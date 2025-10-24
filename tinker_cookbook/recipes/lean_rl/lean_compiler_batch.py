"""
Batch compiler for Lean 4 code compilation in high concurrency scenarios.
This module provides efficient batch processing for RL training with many parallel environments.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any

LEAN_COMPILER_URL = "http://localhost:8965"
LEAN_COMPILER_TIMEOUT = 600


class BatchCompiler:
    """
    Batch compiler for handling multiple Lean compilation requests efficiently.
    This reduces the overhead of individual HTTP requests by batching them together.
    """
    
    def __init__(self, batch_size: int = 32, flush_interval: float = 0.05):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_requests = []
        self._lock = asyncio.Lock()
        self._flush_task = None
    
    async def compile_async(self, code: str, request_id: str) -> dict:
        """
        Add a compilation request to the batch. Returns a future that resolves when compilation is done.
        """
        future = asyncio.Future()
        
        async with self._lock:
            self.pending_requests.append({
                "code": code,
                "request_id": request_id,
                "future": future
            })
            
            # Start flush task if not already running
            if self._flush_task is None:
                self._flush_task = asyncio.create_task(self._flush_batch())
        
        return await future
    
    async def _flush_batch(self):
        """Flush the current batch of requests."""
        await asyncio.sleep(self.flush_interval)
        
        async with self._lock:
            if not self.pending_requests:
                self._flush_task = None
                return
            
            # Take up to batch_size requests
            batch_requests = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]
            
            # Process the batch
            await self._process_batch(batch_requests)
            
            # If there are more requests, schedule another flush
            if self.pending_requests:
                self._flush_task = asyncio.create_task(self._flush_batch())
            else:
                self._flush_task = None
    
    async def _process_batch(self, batch_requests):
        """Process a batch of compilation requests."""
        if not batch_requests:
            return
        
        try:
            # Prepare batch payload
            codes = [req["code"] for req in batch_requests]
            request_ids = [req["request_id"] for req in batch_requests]
            
            # Send batch request to compiler
            batch_results = await self._send_batch_request(codes, request_ids)
            
            # Resolve futures with results
            for i, req in enumerate(batch_requests):
                if i < len(batch_results):
                    req["future"].set_result(batch_results[i])
                else:
                    req["future"].set_exception(Exception("Batch processing failed"))
                    
        except Exception as e:
            # Resolve all futures with exception
            for req in batch_requests:
                req["future"].set_exception(e)
    
    async def _send_batch_request(self, codes: List[str], request_ids: List[str]) -> List[dict]:
        """Send a batch request to the compiler backend."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=LEAN_COMPILER_TIMEOUT)) as session:
            try:
                payload = {
                    "codes": codes,
                    "request_ids": request_ids,
                    "batch_timestamp": asyncio.get_event_loop().time()
                }
                
                async with session.post(
                    f"{LEAN_COMPILER_URL}/api/v1/compile_batch",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("results", [])
                    else:
                        raise Exception(f"Batch compilation failed with status {response.status}")
                        
            except Exception as e:
                # Fallback to individual requests if batch endpoint is not available
                print(f"[BatchCompiler] Batch endpoint failed, falling back to individual requests: {e}")
                return await self._fallback_individual_requests(codes, request_ids)
    
    async def _fallback_individual_requests(self, codes: List[str], request_ids: List[str]) -> List[dict]:
        """Fallback to individual requests if batch endpoint is not available."""
        async def compile_single(code, request_id):
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=LEAN_COMPILER_TIMEOUT)) as session:
                try:
                    payload = {
                        "code": code,
                        "request_id": request_id,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    async with session.post(
                        f"{LEAN_COMPILER_URL}/api/v1/compile_one",
                        json=payload
                    ) as response:
                        result = await response.json()
                        result["_request_id"] = request_id
                        result["_timestamp"] = payload["timestamp"]
                        return result
                except Exception as e:
                    return {
                        "error": str(e),
                        "_request_id": request_id,
                        "_timestamp": asyncio.get_event_loop().time()
                    }
        
        # Process all requests in parallel
        tasks = [compile_single(code, req_id) for code, req_id in zip(codes, request_ids)]
        return await asyncio.gather(*tasks)


# Global batch compiler instance
_batch_compiler = BatchCompiler(batch_size=32, flush_interval=0.05)


async def compile_lean_2_async_batch(code: str, request_id: str) -> dict:
    """
    Use the batch compiler for more efficient compilation in high concurrency scenarios.
    This is the recommended method for RL training with many parallel environments.
    """
    return await _batch_compiler.compile_async(code, request_id)
