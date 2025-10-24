import json
import time
import uuid
import os
import hashlib
import asyncio
import aiohttp


LEAN_COMPILER_REQUEST_DIR = os.path.expanduser("/home/bl3615/data/shared_a/requests")
LEAN_COMPILER_RESPONSE_DIR = os.path.expanduser("/home/bl3615/data/shared_a/responses")
LEAN_COMPILER_TIMEOUT = 600


def generate_name(code, index):
    """Generate a unique name based on the index and hash value of the string"""
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    return f"{index}_{code_hash}"

def compile_lean_1(value_list, timeout=LEAN_COMPILER_TIMEOUT):
    """Node A generates JSON task files and processes the returned results"""
    if not isinstance(value_list, list) or not all(isinstance(v, str) for v in value_list):
        raise ValueError("Input must be a list of strings")

    task_id = str(uuid.uuid4())
    task_data = {
        "task_id": task_id,
        "tasks": [],
        "proof_timeout": 150
    }

    # Generate tasks with unique names
    for i, code in enumerate(value_list):
        task_data["tasks"].append({
            "id": i,
            "name": generate_name(code, i),
            "code": code
        })

    task_file = os.path.join(LEAN_COMPILER_REQUEST_DIR, f"{task_id}.json")
    result = None

    with open(task_file, "w") as f:
        json.dump(task_data, f)
    print(f"[Node A] Submitted task {task_id}, number of tasks: {len(value_list)}")

    start_time = time.time()
    response_file = os.path.join(LEAN_COMPILER_RESPONSE_DIR, f"{task_id}.json")

    results = None

    try:
        while time.time() - start_time < timeout:
            if os.path.exists(response_file):
                time.sleep(1)
                with open(response_file, "r") as f:
                    response_data = json.load(f)

                results = response_data.get("results", [])
                # Sort results based on the index in the name field
                results = sorted(results, key=lambda x: int(x["name"].split('_')[0]))
                print(f"[Node A] Task {task_id} completed")
                break
            time.sleep(1)

        if results is None:
            print(f"[Node A] Task {task_id} timed out without returning!")
    except Exception as e:
        print(f"[Node A] Error occurred while processing task {task_id}: {e}")
        results = None
    finally:
        try:
            if os.path.exists(task_file):
                os.remove(task_file)
                print(f"[Node A] Request file {task_file} deleted")
        except Exception as e:
            print(f"[Node A] Failed to delete request file {task_file}: {e}")

    return results


# async def compile_lean_1_async(value_list, timeout=LEAN_COMPILER_TIMEOUT):
#     """异步版本的Lean编译器调用，不会阻塞事件循环"""
#     if not isinstance(value_list, list) or not all(isinstance(v, str) for v in value_list):
#         raise ValueError("Input must be a list of strings")

#     task_id = str(uuid.uuid4())
#     task_data = {
#         "task_id": task_id,
#         "tasks": [],
#         "proof_timeout": 150
#     }

#     # Generate tasks with unique names
#     for i, code in enumerate(value_list):
#         task_data["tasks"].append({
#             "id": i,
#             "name": generate_name(code, i),
#             "code": code
#         })

#     task_file = os.path.join(LEAN_COMPILER_REQUEST_DIR, f"{task_id}.json")
#     result = None

#     with open(task_file, "w") as f:
#         json.dump(task_data, f)
#     print(f"[Node A] Submitted task {task_id}, number of tasks: {len(value_list)}")

#     start_time = time.time()
#     response_file = os.path.join(LEAN_COMPILER_RESPONSE_DIR, f"{task_id}.json")

#     results = None

#     try:
#         while time.time() - start_time < timeout:
#             if os.path.exists(response_file):
#                 await asyncio.sleep(0.1)  # 使用异步sleep，不阻塞事件循环
#                 with open(response_file, "r") as f:
#                     response_data = json.load(f)

#                 results = response_data.get("results", [])
#                 # Sort results based on the index in the name field
#                 results = sorted(results, key=lambda x: int(x["name"].split('_')[0]))
#                 print(f"[Node A] Task {task_id} completed")
#                 break
#             await asyncio.sleep(0.1)  # 使用异步sleep，不阻塞事件循环

#         if results is None:
#             print(f"[Node A] Task {task_id} timed out without returning!")
#     except Exception as e:
#         print(f"[Node A] Error occurred while processing task {task_id}: {e}")
#         results = None
#     finally:
#         try:
#             if os.path.exists(task_file):
#                 os.remove(task_file)
#                 print(f"[Node A] Request file {task_file} deleted")
#         except Exception as e:
#             print(f"[Node A] Failed to delete request file {task_file}: {e}")

#     return results

# LEAN_COMPILER_URL = "http://localhost:8965"

# async def compile_lean_2_async(value_list, timeout=LEAN_COMPILER_TIMEOUT):
#     if not isinstance(value_list, list) or not all(isinstance(v, str) for v in value_list):
#         raise ValueError("Input must be a list of strings")

#     async def compile_single(code, uid):
#         async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
#             try:
#                 async with session.post(
#                     f"{LEAN_COMPILER_URL}/api/v1/compile_one",
#                     json={"code": code}
#                 ) as response:
#                     result = await response.json()
#                     return uid, result
#             except Exception as e:
#                 return uid, {"error": str(e)}

#     # 并行发送所有请求
#     tasks = [compile_single(code, i) for i, code in enumerate(value_list)]
#     results_with_uid = await asyncio.gather(*tasks)
    
#     # 按原始顺序排序并返回结果
#     results_with_uid.sort(key=lambda x: x[0])
#     results = [result for _, result in results_with_uid]
    
#     return results


# async def compile_lean_2_async_with_request_id(value_list, request_id, timeout=LEAN_COMPILER_TIMEOUT):
#     """
#     Enhanced version with request ID for better tracking and debugging.
#     This helps ensure results are matched to the correct request even in high concurrency.
#     """
#     if not isinstance(value_list, list) or not all(isinstance(v, str) for v in value_list):
#         raise ValueError("Input must be a list of strings")

#     async def compile_single(code, uid):
#         async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
#             try:
#                 # Include request_id in the request for better tracking
#                 payload = {
#                     "code": code,
#                     "request_id": f"{request_id}_{uid}",
#                     "timestamp": asyncio.get_event_loop().time()
#                 }
#                 async with session.post(
#                     f"{LEAN_COMPILER_URL}/api/v1/compile_one",
#                     json=payload
#                 ) as response:
#                     result = await response.json()
#                     # Add request tracking info to result
#                     result["_request_id"] = f"{request_id}_{uid}"
#                     result["_timestamp"] = payload["timestamp"]
#                     return uid, result
#             except Exception as e:
#                 error_result = {
#                     "error": str(e),
#                     "_request_id": f"{request_id}_{uid}",
#                     "_timestamp": asyncio.get_event_loop().time()
#                 }
#                 return uid, error_result

#     # 并行发送所有请求
#     tasks = [compile_single(code, i) for i, code in enumerate(value_list)]
#     results_with_uid = await asyncio.gather(*tasks)
    
#     # 按原始顺序排序并返回结果
#     results_with_uid.sort(key=lambda x: x[0])
#     results = [result for _, result in results_with_uid]
    
#     return results 