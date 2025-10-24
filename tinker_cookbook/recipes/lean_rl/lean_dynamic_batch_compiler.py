"""
Dynamic batch compiler for Lean 4 code compilation.
This module accumulates individual compilation requests from multiple LeanEnv instances
and batches them together to call compile_lean_1 for efficient processing.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import threading

from .lean_compiler import compile_lean_1


@dataclass
class CompilationRequest:
    """Individual compilation request from a LeanEnv instance."""
    code: str
    request_id: str
    env_uid: str
    future: asyncio.Future
    timestamp: float


class DynamicBatchCompiler:
    """
    Dynamic batch compiler that accumulates individual requests and processes them in batches.
    
    Key features:
    1. Accumulates individual code compilation requests from multiple LeanEnv instances
    2. When batch_size (e.g., 100) requests are accumulated, triggers batch processing
    3. Uses compile_lean_1 to process batches, ensuring order preservation
    4. Returns results to individual requesters while maintaining order
    5. Serial batch processing: ensures batches are processed one at a time to prevent out-of-order results
    """
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 60.0):
        """
        Initialize the dynamic batch compiler.
        
        Args:
            batch_size: Number of requests to accumulate before processing (default: 100)
            max_wait_time: Maximum time to wait before processing incomplete batch (default: 60.0s)
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        # Thread-safe request queue
        self._pending_requests: List[CompilationRequest] = []
        self._lock = asyncio.Lock()
        
        # Batch processing lock to ensure serial processing
        self._processing_lock = asyncio.Lock()
        
        # Background task management
        self._flush_task: Optional[asyncio.Task] = None
        self._batch_counter = 0
        
        print(f"[DYNAMIC_BATCH] Initialized with batch_size={batch_size}, max_wait_time={max_wait_time}s")
        print(f"[DYNAMIC_BATCH] Serial processing enabled: batches will be processed one at a time")
    
    async def compile_single_async(self, code: str, request_id: str, env_uid: str) -> Dict[str, Any]:
        """
        Submit a single code compilation request.
        
        Args:
            code: Lean 4 code to compile
            request_id: Unique identifier for this request
            env_uid: Environment instance identifier
            
        Returns:
            Compilation result dictionary
        """
        # Create future for this request
        future = asyncio.Future()
        request = CompilationRequest(
            code=code,
            request_id=request_id,
            env_uid=env_uid,
            future=future,
            timestamp=time.time()
        )
        
        # Add to pending requests
        async with self._lock:
            self._pending_requests.append(request)
            current_count = len(self._pending_requests)
            
            print(f"[DYNAMIC_BATCH] [{env_uid}] Added request {request_id}, queue size: {current_count}/{self.batch_size}")
            
            # Start flush task if not already running
            if self._flush_task is None:
                self._flush_task = asyncio.create_task(self._auto_flush())
            
            # Trigger immediate batch processing if batch is full
            if current_count >= self.batch_size:
                print(f"[DYNAMIC_BATCH] Batch size reached ({current_count}), triggering immediate processing")
                asyncio.create_task(self._process_batch_if_ready())
        
        # Wait for result
        return await future
    
    async def _auto_flush(self):
        """Automatically flush batches based on time or size."""
        while True:
            await asyncio.sleep(self.max_wait_time)
            
            async with self._lock:
                if not self._pending_requests:
                    # No pending requests, stop auto flush task
                    self._flush_task = None
                    print(f"[DYNAMIC_BATCH] Auto-flush task stopped (no pending requests)")
                    break
                
                # Check if oldest request has been waiting too long
                oldest_request = min(self._pending_requests, key=lambda r: r.timestamp)
                wait_time = time.time() - oldest_request.timestamp
                
                if wait_time >= self.max_wait_time:
                    print(f"[DYNAMIC_BATCH] Max wait time exceeded ({wait_time:.2f}s >= {self.max_wait_time}s), processing batch")
                    asyncio.create_task(self._process_batch_if_ready())
    
    async def _process_batch_if_ready(self):
        """Process a batch if there are pending requests. Ensures serial processing."""
        # Acquire processing lock to ensure only one batch is processed at a time
        async with self._processing_lock:
            # Now acquire the pending requests lock
            async with self._lock:
                if not self._pending_requests:
                    return
                
                # Take all pending requests for processing
                batch_requests = self._pending_requests.copy()
                self._pending_requests.clear()
                
                self._batch_counter += 1
                batch_id = self._batch_counter
                
                print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] Acquired processing lock (serial mode)")
            
            # Process the batch while holding the processing lock
            # This ensures batches are processed one at a time
            await self._process_batch(batch_requests, batch_id)
            print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] Released processing lock")
    
    async def _process_batch(self, batch_requests: List[CompilationRequest], batch_id: int):
        """
        Process a batch of compilation requests using compile_lean_1.
        
        Args:
            batch_requests: List of requests to process
            batch_id: Unique identifier for this batch
        """
        batch_size = len(batch_requests)
        print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] Processing batch with {batch_size} requests")
        
        # Extract codes and create mapping for order preservation
        codes = [req.code for req in batch_requests]
        request_map = {i: req for i, req in enumerate(batch_requests)}
        
        # Debug: Print request details
        for i, req in enumerate(batch_requests):
            print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] Request {i}: {req.env_uid} -> {req.request_id}")
        
        try:
            # Call compile_lean_1 with the batch of codes
            print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] Calling compile_lean_1 with {len(codes)} codes...")
            start_time = time.time()
            
            # Run compile_lean_1 in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, compile_lean_1, codes)
            
            processing_time = time.time() - start_time
            print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] compile_lean_1 completed in {processing_time:.2f}s")
            
            if results is None or len(results) == 0:
                print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] ERROR: No results returned from compile_lean_1")
                # Set error for all requests
                for req in batch_requests:
                    if not req.future.done():
                        req.future.set_result({
                            "status": "error",
                            "message": "Batch compilation failed - no results returned",
                            "_request_id": req.request_id,
                            "_batch_id": batch_id
                        })
                return
            
            # Verify result count matches request count
            if len(results) != len(batch_requests):
                print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] ERROR: Result count mismatch! "
                      f"Expected {len(batch_requests)}, got {len(results)}")
                # Set error for all requests
                for req in batch_requests:
                    if not req.future.done():
                        req.future.set_result({
                            "status": "error",
                            "message": f"Batch result count mismatch: expected {len(batch_requests)}, got {len(results)}",
                            "_request_id": req.request_id,
                            "_batch_id": batch_id
                        })
                return
            
            # Distribute results to corresponding requests
            print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] Distributing {len(results)} results to requests...")
            for i, result in enumerate(results):
                if i in request_map:
                    req = request_map[i]
                    
                    # Add metadata to result
                    if isinstance(result, dict):
                        result["_request_id"] = req.request_id
                        result["_batch_id"] = batch_id
                        result["_batch_index"] = i
                        result["_env_uid"] = req.env_uid
                    else:
                        # Wrap non-dict results
                        result = {
                            "result": result,
                            "_request_id": req.request_id,
                            "_batch_id": batch_id,
                            "_batch_index": i,
                            "_env_uid": req.env_uid
                        }
                    
                    # Set result for the future
                    if not req.future.done():
                        req.future.set_result(result)
                        print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] Result {i} -> {req.env_uid} ({req.request_id})")
                    else:
                        print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] WARNING: Future already done for request {i}")
                else:
                    print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] ERROR: No request mapping for result index {i}")
            
            print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] ✅ Batch processing completed successfully")
            
        except Exception as e:
            print(f"[DYNAMIC_BATCH] [BATCH_{batch_id}] ERROR: Batch processing failed: {e}")
            # Set error for all pending requests
            for req in batch_requests:
                if not req.future.done():
                    req.future.set_result({
                        "status": "error",
                        "message": f"Batch processing error: {str(e)}",
                        "_request_id": req.request_id,
                        "_batch_id": batch_id
                    })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics of the batch compiler."""
        return {
            "pending_requests": len(self._pending_requests),
            "batch_size": self.batch_size,
            "max_wait_time": self.max_wait_time,
            "batch_counter": self._batch_counter,
            "flush_task_running": self._flush_task is not None,
            "processing_locked": self._processing_lock.locked()
        }


# Global instance for use across all LeanEnv instances
_global_batch_compiler = DynamicBatchCompiler(batch_size=100, max_wait_time=60.0)


async def compile_single_with_batching(code: str, request_id: str, env_uid: str) -> Dict[str, Any]:
    """
    Compile a single Lean 4 code using dynamic batching.
    
    This is the main interface for LeanEnv instances to use.
    
    Args:
        code: Lean 4 code to compile
        request_id: Unique identifier for this request  
        env_uid: Environment instance identifier
        
    Returns:
        Compilation result dictionary
    """
    return await _global_batch_compiler.compile_single_async(code, request_id, env_uid)


def get_batch_compiler_stats() -> Dict[str, Any]:
    """Get current statistics of the global batch compiler."""
    return _global_batch_compiler.get_stats()
