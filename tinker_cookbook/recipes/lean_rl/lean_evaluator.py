from typing import Any, Callable
import os
import json
import time
import logging
 
import tinker
from tinker import types
 
from tinker_cookbook import renderers
from tinker_cookbook.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.lean_rl.lean_utils_v2 import DeepSeekCoTHandler
from tinker_cookbook.recipes.lean_rl.lean_env import LeanEnv
# from tinker_cookbook.recipes.lean_rl.lean_tool_env_v2 import create_minif2f_dataset_builder, LeanToolRLDataset
import asyncio
from tqdm import tqdm
from typing import Callable
from tinker_cookbook.rl.rollouts import do_group_rollout
from tqdm import tqdm

# Setup logger for this module
logger = logging.getLogger(__name__)

# Import jsave function (assuming it exists in the environment)
# try:
from jload import jsave
# except ImportError:
#     # Fallback implementation if jload is not available
#     def jsave(data, filepath):
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

class MiniF2FTestEvaluator(SamplingClientEvaluator):
    """
    A toy SamplingClientEvaluator that runs a custom evaluation and returns its metrics.
    """
 
    def __init__(
        self, 
        model_name: str,
        renderer_name: str,
        minif2f_dataset_path: str = "/home/zy7019/tinker/tinker-cookbook/tinker_logs/data/minif2f_fixed.jsonl", # default is minif2f_fixed.jsonl
        n: int = 8,
        output_dir: str = "/home/zy7019/tinker/tinker-cookbook/tinker_logs/eval/",
        eval_name: str = "lean_evaluator_output",
        max_tokens: int = 1024*16,
        num_sampling_clients: int = 1,  # 新增：支持多个sampling client
    ):
        """
        Initialize the CustomEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """
        split = 'test'
        handler=DeepSeekCoTHandler()
        initial_data_list = handler.load_split(minif2f_dataset_path, split)

        self.n = n
        self.max_tokens = max_tokens
        self.num_sampling_clients = num_sampling_clients
        self.output_path = os.path.join(output_dir, eval_name)
        self.eval_name = eval_name
        self.tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=self.tokenizer)
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
    
        items_for_llm_processing = []
        
        # Process each original problem once (no n-fold augmentation here)
        for idata_orig in initial_data_list:
            origin_id = idata_orig.get("origin_problem_id", idata_orig.get('problem_id', idata_orig.get('name')))
            if not idata_orig.get("lean4_code"): continue
            
            # Keep original data structure, just add origin_problem_id
            item_for_processing = idata_orig.copy()
            item_for_processing["origin_problem_id"] = origin_id
            items_for_llm_processing.append(item_for_processing)

        if not items_for_llm_processing:
            raise Exception("No data available for LLM processing. Exiting.")

        self.dataset = []
        for i, item_data in enumerate(tqdm(items_for_llm_processing, desc=f"Preparing data...")):
            item_data["lean4_code"] = item_data["lean4_code"].split(":= by")[0] + ":= by sorry"
            # if args.correction_round > 0:
            #     error_str = get_error_str(
            #         item_data.get('compiled_code_that_failed_in_prev_round', ''),
            #         item_data.get('errors_for_compiled_code_from_prev_round', {}).get('errors', []),
            #         args.error_thres
            #     )
            #     prompt_str, messages_for_this = handler.generate_correction_prompt(
            #         lean4_code_original_stmt=item_data["lean4_code"],
            #         history_messages_from_prev_round=item_data.get("history_messages_from_prev_round_for_new_prompt",
            #                                                         []),
            #         prev_round_llm_raw_output=item_data.get("prev_round_llm_raw_output_for_new_prompt", ""),
            #         error_message_for_prev_round=error_str,
            #         tokenizer=hf_tokenizer_for_chat_template,
            #         current_correction_round_num=args.correction_round
            #     )
            # else:  # Initial inference
            prompt_str, messages_for_this = handler.prover_inference(
                item_data["lean4_code"], self.tokenizer
            )
            num_tokens = len(self.tokenizer.tokenize(prompt_str))  
            # num_cot_tokens = len(hf_tokenizer_for_chat_template.tokenize(messages_for_this[1]["content"]))
            self.dataset.append({
                # "cot_token_nums": num_cot_tokens,
                "token_nums": num_tokens,
                "prompts_for_vllm": prompt_str,
                "messages_lists_for_current_prompts": messages_for_this,
                "current_chunk_input_items": item_data
            })

    def grader_fn(self, response: str, target: str) -> bool:
        return target.lower() in response.lower()


    async def _process_single_datum(self, datum: dict, sampling_client: tinker.SamplingClient, sampling_params: types.SamplingParams) -> list[dict]:
        """Process a single datum and return processed records."""
        try:
            prompt_str = datum["prompts_for_vllm"]
            model_input: types.ModelInput = types.ModelInput.from_ints(self.tokenizer.encode(prompt_str, add_special_tokens=True))

            # print(f"Sampling prompt...")
            # Generate n responses for this single problem
            r: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input, num_samples=self.n, sampling_params=sampling_params
            )

            print(f"Sampling response received: {len(r.sequences) if r.sequences else 0} sequences")
            processed_records = []
            origin_id = datum["current_chunk_input_items"]["origin_problem_id"]
            
            # Process each of the n generated responses
            for i in range(self.n):
                tokens: list[int] = r.sequences[i].tokens
                response: renderers.Message = self.renderer.parse_response(tokens)[0]
                llm_response_text = response["content"]
                
                # Create a unique problem_id for each generation
                problem_id = f"{origin_id}_g{i}"
                
                # Process the response following unified_inference.py format
                input_item = datum["current_chunk_input_items"].copy()
                input_item["problem_id"] = problem_id
                input_item["model_input"] = prompt_str
                input_item["messages_history_for_this_attempt"] = datum["messages_lists_for_current_prompts"]
                input_item["model_output"] = llm_response_text
                input_item["id_maps"] = [{"origin_problem_id": origin_id},
                                        {"generation_id": problem_id}]
                
                # Extract code using handler (following unified_inference.py logic)
                handler = DeepSeekCoTHandler()
                extracted_code = handler.extrac_code(llm_response_text)
                if extracted_code == "None" or extracted_code is None:
                    input_item["full_code"] = "None"
                else:
                    input_item["full_code"] = handler.problem_check(input_item["lean4_code"], extracted_code)
                
                processed_records.append(input_item)
            
            print(f"Processed {len(processed_records)} records for origin_id: {origin_id}")
            return processed_records
            
        except Exception as e:
            print(f"Error in _process_single_datum: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run custom evaluation with multiple sampling clients for better performance.
        Args:
            sampling_client: The primary sampling client (used to create additional clients)
        Returns:
            Dictionary of metrics from evaluation
        """
        print(f"🚀 Starting evaluation with {self.num_sampling_clients} sampling clients")
        print(f"📊 Dataset: {len(self.dataset)} examples, {self.n} samples each = {len(self.dataset) * self.n} total generations")
        
        # Create multiple sampling clients
        sampling_clients = [sampling_client]
        for i in range(1, self.num_sampling_clients):
            print(f"Creating sampling client {i+1}/{self.num_sampling_clients}...")
            # Note: In practice, you might need to create additional clients differently
            # For now, we'll use the same client but distribute work
            sampling_clients.append(sampling_client)
        
        return await self._run_multi_client_evaluation(sampling_clients)
    
    async def _run_multi_client_evaluation(self, sampling_clients: list) -> dict[str, float]:
        """Run evaluation with multiple sampling clients"""
        metrics = {}
        all_processed_records = []
        all_inference_code_outputs = []
        
        # Define output file paths
        output_file_path_records = os.path.join(self.output_path, f'full_records.json')
        output_file_path_inference_codes = os.path.join(self.output_path, 'to_inference_codes.json')
        tmp_inference_codes_jsonl = os.path.join(self.output_path, 'tmp_inference_codes.jsonl')
        
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.9,
            top_p=0.9,
            stop=self.renderer.get_stop_sequences(),
        )
        
        start_time = time.time()
        
        # Distribute work across clients using chunk-level fire-and-wait
        print(f"📦 Distributing {len(self.dataset)} examples across {len(sampling_clients)} clients...")
        
        # Calculate chunk size for each client
        chunk_size = max(1, len(self.dataset) // len(sampling_clients))
        chunks = [self.dataset[i:i + chunk_size] for i in range(0, len(self.dataset), chunk_size)]
        
        print(f"📊 Chunk distribution: {[len(chunk) for chunk in chunks]} examples per client")
        
        # Create tasks for each client with their assigned chunk
        all_tasks = []
        for client_idx, (client, chunk) in enumerate(zip(sampling_clients, chunks)):
            if chunk:  # Only create task if chunk is not empty
                task = self._process_chunk_with_client(
                    chunk, client, sampling_params, client_idx
                )
                all_tasks.append(task)
        
        print(f"🔥 Firing {len(all_tasks)} client tasks...")
        send_start_time = time.time()
        
        # Wait for all results with progress tracking
        print("⏳ Waiting for results...")
        results_start_time = time.time()
        
        # Open JSONL file for real-time writing
        with open(tmp_inference_codes_jsonl, 'w', encoding='utf-8') as jsonl_file:
            completed_tasks = 0
            for coro in tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc="Processing client chunks"):
                try:
                    client_idx, batch_results = await coro
                    completed_tasks += 1
                    print(f"✅ Client {client_idx} completed ({completed_tasks}/{len(all_tasks)})")
                    
                    if not batch_results:
                        print(f"⚠️  Warning: Client {client_idx} returned empty results!")
                        continue
                    
                    # Process results as they come in
                    for input_item in batch_results:
                        all_processed_records.append(input_item)
                        
                        # Create inference code output record
                        inference_record = {
                            "problem_id": input_item["problem_id"],
                            "origin_problem_id": input_item.get("origin_problem_id"),
                            "id_maps": input_item.get("id_maps"),
                            "lean4_code": input_item["lean4_code"],
                            "model_input": input_item["model_input"],
                            "messages_history_list": input_item["messages_history_for_this_attempt"],
                            "model_output": input_item["model_output"],
                            "full_code": input_item["full_code"]
                        }
                        
                        # Add to in-memory list
                        all_inference_code_outputs.append(inference_record)
                        
                        # Write to JSONL file immediately
                        jsonl_file.write(json.dumps(inference_record, ensure_ascii=False) + '\n')
                        jsonl_file.flush()
                        
                except Exception as e:
                    print(f"❌ Client task failed: {e}")
                    import traceback
                    print(traceback.format_exc())
        
        send_time = time.time() - send_start_time
        results_time = time.time() - results_start_time
        total_time = time.time() - start_time
        
        print(f"📊 Performance Summary:")
        print(f"  Send time: {send_time:.2f}s")
        print(f"  Results time: {results_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {len(self.dataset)/total_time:.2f} examples/second")
        
        # Save results
        print(f"💾 Saving results to {self.output_path}")
        jsave(all_processed_records, output_file_path_records)
        jsave(all_inference_code_outputs, output_file_path_inference_codes)
        
        # Calculate metrics
        total_examples = len(self.dataset)
        total_generations = total_examples * self.n
        
        metrics[f"{self.eval_name}/total_examples"] = total_examples
        metrics[f"{self.eval_name}/total_generations"] = total_generations
        metrics[f"{self.eval_name}/generations_per_example"] = self.n
        metrics[f"{self.eval_name}/num_sampling_clients"] = self.num_sampling_clients
        metrics[f"{self.eval_name}/total_processing_time"] = total_time
        metrics[f"{self.eval_name}/avg_time_per_example"] = total_time / total_examples if total_examples > 0 else 0.0
        metrics[f"{self.eval_name}/throughput_examples_per_sec"] = total_examples / total_time if total_time > 0 else 0.0
        
        # Add detailed performance metrics
        metrics[f"{self.eval_name}/send_time"] = send_time
        metrics[f"{self.eval_name}/results_time"] = results_time
        metrics[f"{self.eval_name}/avg_time_per_generation"] = total_time / total_generations if total_generations > 0 else 0.0
        
        # Log structured metrics
        print(f"🎉 Evaluation completed. Key metrics:")
        print(f"  Total examples: {total_examples}")
        print(f"  Total generations: {total_generations}")
        print(f"  Processing time: {total_time:.2f}s")
        print(f"  Throughput: {total_examples/total_time:.2f} examples/sec")
        print(f"  All metrics: {metrics}")
        
        return metrics
    
    async def _process_chunk_with_client(
        self, 
        chunk: list, 
        sampling_client: tinker.SamplingClient, 
        sampling_params: types.SamplingParams,
        client_idx: int
    ) -> tuple[int, list]:
        """Process a chunk of data with a specific sampling client"""
        print(f"🔄 Client {client_idx} processing {len(chunk)} examples...")
        
        # Create tasks for this chunk
        chunk_tasks = [
            self._process_single_datum(datum, sampling_client, sampling_params)
            for datum in chunk
        ]
        
        # Process chunk with fire-and-wait strategy
        chunk_results = []
        for coro in asyncio.as_completed(chunk_tasks):
            try:
                batch_result = await coro
                if batch_result:
                    chunk_results.extend(batch_result)
            except Exception as e:
                print(f"❌ Client {client_idx} datum failed: {e}")
        
        print(f"✅ Client {client_idx} completed {len(chunk_results)} results")
        return client_idx, chunk_results


    async def call_w_leanrl(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run custom evaluation on the given sampling client and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics from inspect evaluation
        """
 
        metrics = {}
 
        num_examples = len(self.dataset)
        num_correct = 0
 
        sampling_params = types.SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )
 
        for datum in tqdm(self.dataset, desc="Evaluating examples"):
            prompt_str = datum["prompts_for_vllm"]
            model_input: types.ModelInput = types.ModelInput.from_ints(self.tokenizer.encode(prompt_str, add_special_tokens=True))
            lean_env = LeanEnv(
                problem=prompt_str,
                renderer=self.renderer,
            )
            # Generate response
            r: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input, num_samples=self.n, sampling_params=sampling_params
            )
            # tokens: list[int] = r.sequences[0].tokens

            for i in range(self.n):
                tokens: list[int] = r.sequences[i].tokens
                response: renderers.Message = self.renderer.parse_response(tokens)[0]
                full_code = prompt_str + response["content"]
                answer_reward = lean_env.check_answer(full_code)
                if answer_reward > 0:
                    num_correct += 1
                    print(f"correct")
                else:
                    print(f"wrong")
                # if self.grader_fn(response["content"], datum["current_chunk_input_items"]["lean4_code"]):
                    # num_correct += 1
        metrics["accuracy"] = num_correct / num_examples
        return metrics

def create_minif2f_evaluator_builder(
    dataset_path: str,
    model_name: str,
    renderer_name: str,
    eval_name: str = "minif2f_eval",
    n: int = 8,
    max_tokens: int = 1024 * 16,
    output_dir: str = "/home/zy7019/tinker/tinker-cookbook/tinker_logs/eval/",
    num_sampling_clients: int = 1,
) -> Callable[[], MiniF2FTestEvaluator]:
    """
    创建一个MiniF2F评估器构建器函数，用于在监督学习训练中集成评估器。
    
    Args:
        dataset_path: MiniF2F数据集路径
        model_name: 模型名称
        renderer_name: 渲染器名称
        eval_name: 评估器名称
        n: 每个问题生成的样本数
        max_tokens: 最大token数
        output_dir: 输出目录
        num_sampling_clients: 采样客户端数量
    
    Returns:
        返回一个构建器函数，该函数返回MiniF2FTestEvaluator实例
    """
    def builder() -> MiniF2FTestEvaluator:
        return MiniF2FTestEvaluator(
            model_name=model_name,
            renderer_name=renderer_name,
            minif2f_dataset_path=dataset_path,
            n=n,
            output_dir=output_dir,
            eval_name=eval_name,
            max_tokens=max_tokens,
            num_sampling_clients=num_sampling_clients,
        )
    
    return builder


class Minif2FLeanToolEvaluator(SamplingClientEvaluator):
    """
    Evaluator for minif2f dataset using LeanToolEnv with do_group_rollout interface.
    This evaluator performs multi-round interactions and computes pass@k statistics.
    """
    
    def __init__(
        self,
        minif2f_dataset_path: str,
        model_name: str = "Qwen/Qwen3-30B-A3B",
        renderer_name: str = "qwen",
        max_rounds: int = 10,
        max_tokens: int = 1024 * 16,
        group_size: int = 1,  # Number of rollouts per problem
        k_values: list[int] = [1, 8, 32],
        # Reward coefficients for evaluation (focus on final answer)
        tool_format_coef: float = 0.0,  # Set to 0 to focus on answer correctness
        tool_answer_coef: float = 1.0,
        lemma_reward_coef: float = 0.0,  # Set to 0 to focus on final answer
        final_reward_coef: float = 1.0,
        mode: str = "normal",
        enable_thinking: bool = False,
        chunk_num: int = 10,
    ):
        self.minif2f_dataset_path = minif2f_dataset_path
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.group_size = group_size
        self.k_values = k_values
        self.tool_format_coef = tool_format_coef
        self.tool_answer_coef = tool_answer_coef
        self.lemma_reward_coef = lemma_reward_coef
        self.final_reward_coef = final_reward_coef
        self.mode = mode
        self.enable_thinking = enable_thinking
        self.chunk_num = chunk_num
        
        # Load dataset
        self.dataset = self._load_dataset()
        
    def _load_dataset(self):
        """Load minif2f dataset using the dataset builder."""
        try:
            print(f"📂 Loading minif2f dataset from: {self.minif2f_dataset_path}")
            dataset_builder = create_minif2f_dataset_builder(
                train_dataset_path=self.minif2f_dataset_path,
                test_dataset_path=None,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                batch_size=1,  # Process one problem at a time
                group_size=self.group_size,
                max_rounds=self.max_rounds,
                mode=self.mode,
                enable_thinking=self.enable_thinking,
                tool_format_coef=self.tool_format_coef,
                tool_answer_coef=self.tool_answer_coef,
                lemma_reward_coef=self.lemma_reward_coef,
                final_reward_coef=self.final_reward_coef,
            )
            
            train_ds, _ = dataset_builder()
            print(f"✅ Successfully loaded dataset with {len(train_ds)} problems")
            return train_ds
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print(f"📁 Dataset path: {self.minif2f_dataset_path}")
            print(f"🔍 Please check if the file exists and has valid JSON format")
            raise
    
    def _split_dataset_into_chunks(self, dataset):
        """Split dataset into chunks for processing."""
        total_problems = len(dataset)
        chunk_size = max(1, total_problems // self.chunk_num)
        
        # Instead of creating new LeanToolRLDataset objects, we'll work with batch indices
        # and create a simple chunk representation
        chunks = []
        for i in range(0, total_problems, chunk_size):
            chunk_end = min(i + chunk_size, total_problems)
            # Store chunk information: (start_batch_idx, end_batch_idx, original_dataset)
            chunk_info = {
                'start_batch_idx': i,
                'end_batch_idx': chunk_end,
                'dataset': dataset,
                'size': chunk_end - i
            }
            chunks.append(chunk_info)
        
        print(f"📊 Split {total_problems} problems into {len(chunks)} chunks")
        print(f"📦 Chunk sizes: {[chunk['size'] for chunk in chunks]}")
        
        return chunks
    
    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Evaluate the model on minif2f dataset using do_group_rollout with chunk-based processing.
        
        Args:
            sampling_client: The sampling client to evaluate
            
        Returns:
            Dictionary of evaluation metrics including pass@k statistics
        """
        from tinker_cookbook.completers import TinkerTokenCompleter
        from tinker_cookbook.rl.rollouts import do_group_rollout
        
        print(f"🚀 Starting minif2f evaluation with {len(self.dataset)} problems")
        print(f"📊 Each problem will have {self.group_size} rollouts")
        print(f"🎯 Computing pass@k for k={self.k_values}")
        print(f"📦 Processing in {self.chunk_num} chunks")
        
        # Create policy
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        
        # Split dataset into chunks
        chunks = self._split_dataset_into_chunks(self.dataset)
        
        # Collect all results
        all_problem_results = []
        problem_pass_results = {}  # problem_id -> list of pass/fail results
        
        # Process each chunk sequentially, but within each chunk use fire-and-wait
        for chunk_idx, chunk_info in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
            print(f"\n🔄 Processing chunk {chunk_idx + 1}/{len(chunks)} ({chunk_info['size']} problems)")
            
            # Create tasks for all problems in this chunk using fire-and-wait algorithm
            print(f"🔥 Firing {chunk_info['size']} problem tasks in chunk {chunk_idx + 1}...")
            chunk_tasks = []
            
            # Process each batch in this chunk
            for local_batch_idx in range(chunk_info['start_batch_idx'], chunk_info['end_batch_idx']):
                # Get environment group builders for this problem
                env_group_builders = chunk_info['dataset'].get_batch(local_batch_idx)
                
                if not env_group_builders:
                    print(f"⚠️  No environment group builders for batch {local_batch_idx} in chunk {chunk_idx}")
                    continue
                
                # Create task for this problem
                task = self._process_single_problem(
                    local_batch_idx, env_group_builders, policy
                )
                chunk_tasks.append((local_batch_idx, task))
            
            print(f"⏳ Waiting for {len(chunk_tasks)} problem results in chunk {chunk_idx + 1}...")
            
            # Process results as they complete within this chunk
            for coro in asyncio.as_completed([task for _, task in chunk_tasks]):
                try:
                    result_dict = await coro
                    batch_idx = result_dict["batch_idx"]
                    problem_results = result_dict["trajectory_groups"]
                    print(f"✅ Problem {batch_idx + 1}/{len(self.dataset)} completed (chunk {chunk_idx + 1})")
                    
                    # Process results for this problem
                    problem_id = f"problem_{batch_idx}"
                    problem_pass_results[problem_id] = []
                    
                    for group_idx, traj_group in enumerate(problem_results):
                        for traj_idx, traj in enumerate(traj_group.trajectories_G):
                            # Check if the last transition has positive reward (answer correct)
                            if traj.transitions:
                                last_transition = traj.transitions[-1]
                                # The reward should be > 0 if the final answer is correct
                                # Since we set format_coef=0 and lemma_coef=0, only answer_coef matters
                                is_correct = last_transition.reward > 0
                                problem_pass_results[problem_id].append(is_correct)
                                
                                # Store detailed results
                                result = {
                                    "problem_id": problem_id,
                                    "group_idx": group_idx,
                                    "traj_idx": traj_idx,
                                    "is_correct": is_correct,
                                    "total_reward": sum(t.reward for t in traj.transitions),
                                    "final_reward": last_transition.reward,
                                    "num_transitions": len(traj.transitions),
                                    "episode_done": last_transition.episode_done,
                                    "metrics": last_transition.metrics,
                                    "all_transition_rewards": [t.reward for t in traj.transitions],
                                }
                                all_problem_results.append(result)
                                
                                print(f"  Problem {batch_idx + 1} Rollout {traj_idx + 1}: {'✅ PASS' if is_correct else '❌ FAIL'} "
                                      f"(reward: {last_transition.reward:.3f}, steps: {len(traj.transitions)})")
                    
                except Exception as e:
                    print(f"❌ Problem task failed in chunk {chunk_idx + 1}: {e}")
                    import traceback
                    print(traceback.format_exc())
            
            print(f"✅ Chunk {chunk_idx + 1} completed")
        
        # Sort results by problem index to maintain original order
        all_problem_results.sort(key=lambda x: int(x["problem_id"].split("_")[1]))
        
        # Compute pass@k statistics
        metrics = self._compute_pass_at_k_metrics(problem_pass_results)
        
        # Add detailed trajectory statistics
        trajectory_metrics = self._compute_trajectory_metrics(all_problem_results)
        metrics.update(trajectory_metrics)
        
        # Add summary statistics
        summary_metrics = self._compute_summary_metrics(all_problem_results)
        metrics.update(summary_metrics)
        
        # Add evaluation configuration to metrics
        metrics["evaluation_config/max_rounds"] = self.max_rounds
        metrics["evaluation_config/group_size"] = self.group_size
        metrics["evaluation_config/k_values"] = self.k_values
        metrics["evaluation_config/chunk_num"] = self.chunk_num
        
        print(f"\n🎉 Evaluation completed!")
        print(f"📊 Results summary:")
        for k in self.k_values:
            if f"pass_at_{k}" in metrics:
                if metrics[f"pass_at_{k}"] is not None:
                    print(f"  Pass@{k}: {metrics[f'pass_at_{k}']:.3f}")
                else:
                    print(f"  Pass@{k}: N/A ({metrics.get(f'pass_at_{k}_note', 'insufficient rollouts')})")
        print(f"  Total problems: {metrics['total_problems']}")
        print(f"  Total rollouts: {metrics['total_rollouts']}")
        print(f"  Overall pass rate: {metrics['overall_pass_rate']:.3f}")
        print(f"  All metrics: {metrics}")
        
        return metrics
    
    def _compute_pass_at_k_metrics(self, problem_pass_results: dict[str, list[bool]]) -> dict[str, float]:
        """Compute pass@k metrics for each k value."""
        metrics = {}
        
        # Check if we have enough rollouts for all k values
        max_rollouts_per_problem = max(len(pass_results) for pass_results in problem_pass_results.values()) if problem_pass_results else 0
        
        for k in self.k_values:
            if max_rollouts_per_problem < k:
                print(f"⚠️  Warning: group_size ({max_rollouts_per_problem}) < k ({k}). Pass@{k} may not be meaningful.")
                # Option 1: Skip this metric
                # continue
                
                # Option 2: Use available rollouts but mark as potentially unreliable
                metrics[f"pass_at_{k}"] = None
                metrics[f"pass_at_{k}_note"] = f"Only {max_rollouts_per_problem} rollouts available, result may be unreliable"
                continue
            
            pass_at_k_scores = []
            
            for problem_id, pass_results in problem_pass_results.items():
                if len(pass_results) >= k:
                    # Take the first k results and check if any is correct
                    k_results = pass_results[:k]
                    pass_at_k = any(k_results)
                    pass_at_k_scores.append(pass_at_k)
                else:
                    # This should not happen if we checked above, but handle gracefully
                    if pass_results:
                        pass_at_k = any(pass_results)
                        pass_at_k_scores.append(pass_at_k)
            
            if pass_at_k_scores:
                metrics[f"pass_at_{k}"] = sum(pass_at_k_scores) / len(pass_at_k_scores)
            else:
                metrics[f"pass_at_{k}"] = 0.0
        
        return metrics
    
    def _compute_trajectory_metrics(self, all_results: list[dict]) -> dict[str, float]:
        """Compute detailed trajectory statistics."""
        if not all_results:
            return {}
        
        # Basic statistics
        total_rollouts = len(all_results)
        correct_rollouts = sum(1 for r in all_results if r["is_correct"])
        
        # Reward statistics
        all_rewards = [r["total_reward"] for r in all_results]
        final_rewards = [r["final_reward"] for r in all_results]
        
        # Step statistics
        all_steps = [r["num_transitions"] for r in all_results]
        
        # Response type statistics (from metrics)
        response_type_counts = {"tool_call": 0, "truncated": 0, "im_end": 0}
        for result in all_results:
            if "metrics" in result and result["metrics"]:
                response_counts = result["metrics"].get("response_type_counts", {})
                for key in response_type_counts:
                    response_type_counts[key] += response_counts.get(key, 0)
        
        metrics = {
            "total_rollouts": total_rollouts,
            "correct_rollouts": correct_rollouts,
            "overall_pass_rate": correct_rollouts / total_rollouts if total_rollouts > 0 else 0.0,
            
            # Reward statistics
            "avg_total_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "avg_final_reward": sum(final_rewards) / len(final_rewards) if final_rewards else 0.0,
            "max_total_reward": max(all_rewards) if all_rewards else 0.0,
            "max_final_reward": max(final_rewards) if final_rewards else 0.0,
            
            # Step statistics
            "avg_steps": sum(all_steps) / len(all_steps) if all_steps else 0.0,
            "max_steps": max(all_steps) if all_steps else 0.0,
            "min_steps": min(all_steps) if all_steps else 0.0,
            
            # Response type statistics
            "total_tool_calls": response_type_counts["tool_call"],
            "total_truncated": response_type_counts["truncated"],
            "total_im_end": response_type_counts["im_end"],
        }
        
        # Add response type ratios
        total_responses = sum(response_type_counts.values())
        if total_responses > 0:
            metrics["tool_call_ratio"] = response_type_counts["tool_call"] / total_responses
            metrics["truncated_ratio"] = response_type_counts["truncated"] / total_responses
            metrics["im_end_ratio"] = response_type_counts["im_end"] / total_responses
        
        return metrics
    
    def _compute_summary_metrics(self, all_results: list[dict]) -> dict[str, float]:
        """Compute summary metrics."""
        if not all_results:
            return {"total_problems": 0, "total_rollouts": 0}
        
        # Count unique problems
        unique_problems = set(r["problem_id"] for r in all_results)
        
        return {
            "total_problems": len(unique_problems),
            "total_rollouts": len(all_results),
        }
    
    async def _process_single_problem(self, batch_idx: int, env_group_builders, policy):
        """
        Process a single problem with its environment group builders.
        
        Args:
            batch_idx: Index of the problem batch
            env_group_builders: List of environment group builders for this problem
            policy: The policy to use for rollouts
            
        Returns:
            Dictionary containing batch_idx and trajectory_groups
        """
        
        # Perform rollouts for this problem
        trajectory_groups = await asyncio.gather(
            *[do_group_rollout(builder, policy) for builder in env_group_builders]
        )
        
        return {
            "batch_idx": batch_idx,
            "trajectory_groups": trajectory_groups
        }


def create_minif2f_lean_tool_evaluator_builder(
    minif2f_dataset_path: str,
    model_name: str = "Qwen/Qwen3-30B-A3B",
    renderer_name: str = "qwen",
    max_rounds: int = 10,
    max_tokens: int = 1024 * 16,
    group_size: int = 1,
    k_values: list[int] = [1, 8, 32],
    tool_format_coef: float = 0.0,
    tool_answer_coef: float = 1.0,
    lemma_reward_coef: float = 0.0,
    final_reward_coef: float = 1.0,
    mode: str = "normal",
    enable_thinking: bool = False,
    chunk_num: int = 10,
) -> Callable[[], Minif2FLeanToolEvaluator]:
    """
    创建一个 minif2f LeanTool 评估器构建器函数。
    
    Args:
        minif2f_dataset_path: minif2f 数据集路径
        model_name: 模型名称
        renderer_name: 渲染器名称
        max_rounds: 最大轮数
        max_tokens: 最大token数
        group_size: 每个问题的rollout数量
        k_values: 要计算的pass@k值列表
        tool_format_coef: 工具格式系数
        tool_answer_coef: 工具答案系数
        lemma_reward_coef: 引理奖励系数
        final_reward_coef: 最终奖励系数
        mode: 模式
        enable_thinking: 是否启用思考
        chunk_num: 数据集分块数量，默认为10
    
    Returns:
        返回一个构建器函数，该函数返回Minif2FLeanToolEvaluator实例
    """
    def builder() -> Minif2FLeanToolEvaluator:
        return Minif2FLeanToolEvaluator(
            minif2f_dataset_path=minif2f_dataset_path,
            model_name=model_name,
            renderer_name=renderer_name,
            max_rounds=max_rounds,
            max_tokens=max_tokens,
            group_size=group_size,
            k_values=k_values,
            tool_format_coef=tool_format_coef,
            tool_answer_coef=tool_answer_coef,
            lemma_reward_coef=lemma_reward_coef,
            final_reward_coef=final_reward_coef,
            mode=mode,
            enable_thinking=enable_thinking,
            chunk_num=chunk_num,
        )
    
    return builder


if __name__ == "__main__":
    QA_DATASET = [
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "What is the capital of Germany?", "output": "Berlin"},
        {"input": "What is the capital of Italy?", "output": "Rome"},
    ]
    
    def grader_fn(response: str, target: str) -> bool:
        return target.lower() in response.lower()
    
    model_name =  "Qwen/Qwen3-30B-A3B"  
    # Use a default renderer name since model_info is not available
    renderer_name = "qwen"  # or whatever the appropriate default is
    evaluator = MiniF2FTestEvaluator(
        model_name=model_name,
        renderer_name=renderer_name,
        
    )
    
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model="meta-llama/Llama-3.1-8B-Instruct")
    
    async def main():
        result = await evaluator(sampling_client)
        print(result)
    
    asyncio.run(main())



    