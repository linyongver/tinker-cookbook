"""
Lean 4 environment for RL training.

This environment uses a remote compiler endpoint to verify Lean 4 code correctness.
"""
import asyncio
from functools import partial
from typing import Literal
import time
import random

import chz
import uuid
from datasets import load_dataset
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder, Action, StepResult, Observation, StopCondition
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tinker_cookbook.recipes.lean_rl.lean_utils_v2 import DeepSeekCoTHandler as LeanCodeHandler
from tinker_cookbook.recipes.lean_rl.lean_utils_v2 import remove_comments, remove_specific_lines, add_header
# from tinker_cookbook.recipes.lean_rl.lean_compiler import compile_lean_1, compile_lean_1_async, compile_lean_2_async, compile_lean_2_async_with_request_id
from tinker_cookbook.recipes.lean_rl.lean_dynamic_batch_compiler import compile_single_with_batching, get_batch_compiler_stats

class LeanEnv(ProblemEnv):
    """
    Environment for Lean 4 theorem proving.
    
    The model is given a Lean 4 theorem statement and must provide a complete proof.
    The reward is based on whether the proof compiles and is complete.
    """
    
    def __init__(
        self,
        chat_history, # chat history
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.0,
        max_rounds: int = 1,
    ):
        # Generate unique identifier for this environment instance
        self.uid = str(uuid.uuid4())[:8]  # Use first 8 characters for brevity
        
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.format_coef = format_coef
        self.chat_history = chat_history
        self.max_rounds = max_rounds
        self.handler = LeanCodeHandler()
        
        # Track conversation state
        self.step_count = 0
        self.episode_done = False
        self.current_conversation = None
    
    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return initial observation and stop condition."""
        self.step_count = 0  # Reset step counter
        self.episode_done = False  # Reset episode done status
        self.current_conversation = self.chat_history.copy()
        
        initial_obs = self.renderer.build_generation_prompt(self.current_conversation)
        
        print(f"[{self.uid}] [INIT] {initial_obs.length}t (max_rounds: {self.max_rounds})")
        
        return initial_obs, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Process a step in the environment."""
        self.step_count += 1
        
        # Check if episode should end due to max rounds
        max_rounds_reached = self.step_count >= self.max_rounds
        
        # Parse the action
        message, parse_success = self.renderer.parse_response(action)
        
        # Add message to conversation
        self.current_conversation.append(message)
        
        # Check format and answer
        new_convo_str = self.renderer.tokenizer.apply_chat_template(
            self.current_conversation[:1]+self.current_conversation[-1:], tokenize=False, add_generation_prompt=False
        )
        correct_format, code_to_compile = self._check_format_with_code(new_convo_str)
        correct_answer, compile_result = await self.check_answer_async_with_result(code_to_compile)
        
        # Calculate token count
        action_tokens = len(action)
        
        # Determine episode end reason for logging
        episode_end_reason = ""
        if max_rounds_reached:
            episode_end_reason = " [MAX_ROUNDS]"
        elif correct_answer:
            episode_end_reason = " [CORRECT_ANSWER]"
        
        # Episode ends if either max rounds reached OR answer is correct
        self.episode_done = max_rounds_reached or correct_answer
        
        # Calculate reward
        total_reward = self.format_coef * correct_format + correct_answer
        
        # Add appropriate user response based on result
        if correct_answer:
            # Answer is correct - add success message
            self.current_conversation.append(
                renderers.Message(role="user", content="The Answer is Correct")
            )
            print(f"[{self.uid}] [S{self.step_count}] CORRECT ({action_tokens}t){episode_end_reason}")
        else:
            # Answer is wrong - add error message with compilation result
            error_message = self._format_error_response(compile_result)
            self.current_conversation.append(error_message)
            
            if max_rounds_reached:
                print(f"[{self.uid}] [S{self.step_count}] WRONG_MAX_ROUNDS ({action_tokens}t){episode_end_reason}")
            else:
                print(f"[{self.uid}] [S{self.step_count}] WRONG_CONTINUE ({action_tokens}t)")
        
        # Calculate reward
        if correct_answer:
            reward = 1.0  # Always give reward 1 for correct answer
        else:
            reward = total_reward
        
        # Build next observation
        next_observation = self.renderer.build_generation_prompt(self.current_conversation)
        
        return StepResult(
            reward=reward,
            episode_done=self.episode_done,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
                "step_count": self.step_count,
                "max_rounds_reached": max_rounds_reached,
            },
        )
    
    def get_question(self) -> str:
        """Return the question/problem statement."""
        # Extract the last user message from chat history if available
        for msg in reversed(self.chat_history):
            if msg.get("role") == "user":
                return msg.get("content", "")
        # Fallback: return empty string if no user message found
        return ""
    
    def check_answer(self, sample_str: str) -> bool:
        """Synchronous check - not used in async step() but required by abstract base class."""
        # Since this environment uses async checking in step(), this is a placeholder
        # In practice, the async version check_answer_async_with_result is used
        try:
            return self.check_format(sample_str)
        except Exception:
            return False
    
    # async def check_answer_async(self, sample_str: str) -> bool:
    #     """异步版本：检查答案是否正确，返回编译结果字典"""
    #     correct, _ = await self.check_answer_async_with_result(sample_str)
    #     return correct
    
    async def check_answer_async_with_result(self, sample_str: str) -> tuple[bool, dict]:
        """异步版本：检查答案是否正确，返回(是否正确, 编译结果)"""
        # Use the same compiler interface as lean_tool_env_v2
        code = remove_comments(sample_str)
        # code = add_header(code)
        code = code.strip()
        
        # Generate unique request ID with additional entropy for better uniqueness
        timestamp = int(time.time() * 1000000)  # 微秒时间戳
        random_suffix = random.randint(1000, 9999)  # 4位随机数
        request_id = f"{self.uid}_{self.step_count}_{timestamp}_{random_suffix}"
        
        # Use dynamic batch compiler - single code will be batched with others
        print(f"[{self.uid}] [DYNAMIC_BATCH] Submitting code with request_id={request_id}")
        
        # Get current batch stats for debugging
        stats = get_batch_compiler_stats()
        print(f"[{self.uid}] [DYNAMIC_BATCH] Current batch stats: {stats}")
        
        compile_result = await compile_single_with_batching(code, request_id, self.uid)
        print(f"[{self.uid}] [DYNAMIC_BATCH] Got result for request_id={request_id}")
        
        # Verify request ID matches (safety check)
        if "_request_id" in compile_result:
            actual_request_id = compile_result["_request_id"]
            if actual_request_id != request_id:
                print(f"[{self.uid}] [ERROR] CRITICAL: Request ID mismatch!")
                print(f"[{self.uid}] [ERROR] Expected: {request_id}")
                print(f"[{self.uid}] [ERROR] Actual: {actual_request_id}")
                print(f"[{self.uid}] [ERROR] This indicates a batch processing error!")
                # Return error to prevent training on wrong results
                return False, {"status": "error", "message": f"Request ID mismatch: expected {request_id}, got {actual_request_id}"}
            else:
                print(f"[{self.uid}] [DYNAMIC_BATCH] Request ID verified: {actual_request_id}")
        else:
            print(f"[{self.uid}] [WARNING] No request ID in result, cannot verify correctness")
        
        # Check for batch processing errors
        if compile_result.get("status") == "error":
            return False, compile_result
        
        # compile_result 是一个字典，不是列表
        if not isinstance(compile_result, dict) or 'compilation_result' not in compile_result:
            return False, {"status": "error", "message": f"Invalid compile result: {compile_result}"}
        comp_result = compile_result['compilation_result']
        is_pass = comp_result.get('pass', False)
        is_complete = comp_result.get('complete', False)
        
        # Return the full compile result for error formatting
        return is_pass and is_complete, compile_result
        
    def _check_format_with_code(self, sample_str: str) -> tuple[bool, str]:
        """
        Internal method: Check if the response contains valid Lean 4 code.
        Returns (format_correct, code_to_compile)
        """
        try:
            solution_str = remove_comments(sample_str)
            full_proof = self.handler.extrac_code(solution_str)
            original_lean4_code_in_input = self.handler.extract_original_lean4_code(solution_str)
            full_code = self.handler.problem_check(original_lean4_code_in_input, full_proof)
            code_to_compile = remove_specific_lines(full_code).strip()
        except Exception as e:
            print(f"Error in compile_lean_code: {e}")
            return False, "None"
        if "[[Error]]" in full_code:
            return False, "None"
        return True, code_to_compile
    
    def check_format(self, sample_str: str) -> bool:
        """
        Check if the response contains valid Lean 4 code.
        Returns True if format is correct, False otherwise.
        """
        format_ok, _ = self._check_format_with_code(sample_str)
        return format_ok
    
    def _format_error_response(self, compile_result: dict) -> renderers.Message:
        """Format error response with compilation result."""
        try:
            if isinstance(compile_result, dict) and 'compilation_result' in compile_result:
                comp_result = compile_result['compilation_result'].copy()
                # Remove verbose warnings
                comp_result.pop('warnings', None)
                error_msg = str(comp_result)
            else:
                error_msg = f"Compilation failed - invalid result format, {str(compile_result)}"
        except Exception as e:
            error_msg = f"Compilation error: {str(e)}"
        
        content = f"The code is incorrect. Error message: {error_msg}. Please correct the code and try again."
        return renderers.Message(role="user", content=content)
    
    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        # for now just return empty list
        return []




class LeanRLDataset(RLDataset):
    """
    Dataset for Lean 4 theorem proving problems.
    """
    
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        # split: Literal["train", "test"] = "train",
        dataset_path: str | None = None,
        max_rounds: int = 1,
        format_coef: float = 0.0,
    ):
        if dataset_path:
            self.ds = load_dataset('json', data_files=dataset_path)
            self.ds = self.ds['train']
        else:
            raise ValueError("dataset_path is required")
            # # Use default sample dataset as fallback
            # self.ds = self._create_sample_dataset()
            
        self.ds = self.ds.shuffle(seed=0)
        
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.max_rounds = max_rounds
        self.format_coef = format_coef

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        return [
            builder
            for row in self.ds.select(range(index * self.batch_size, (index + 1) * self.batch_size))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None
        ]

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size
    
    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            chat_history = x.get("messages", [])
            if not chat_history:
                logger.warning(f"No chat history found in dataset row: {x}")
                return None
                
            return ProblemGroupBuilder(
                env_thunk=partial(
                    LeanEnv, chat_history, self.renderer, convo_prefix=self.convo_prefix, max_rounds=self.max_rounds, format_coef=self.format_coef
                ),
                num_envs=group_size,
                dataset_name="leanrl",
            )
        except Exception as e:
            logger.warning(f"Failed to create env group builder: {e}")
            return None


# from tinker_cookbook.recipes.lean_rl.lean_utils_v2 import DeepSeekCoTHandler
# class MiniF2FTestDataset(LeanRLDataset):
#     def __init__(
#         self,
#         batch_size: int,
#         group_size: int,
#         renderer: renderers.Renderer,
#         convo_prefix: list[renderers.Message] | None = None,
#         split: Literal["train", "test"] = "train",
#         dataset_path: str | None = None,  # Add dataset path parameter
#     ):
#         self.handler=DeepSeekCoTHandler()
#         self.split = 'test'
#         initial_data_list = self.handler.load_split(dataset_path, self.split)
#         self.ds = Dataset.from_list(initial_data_list)
#         self.batch_size = batch_size
#         self.group_size = group_size
#         self.renderer = renderer
#         self.convo_prefix = convo_prefix
        



@chz.chz
class LeanRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    train_dataset_path: str | None = None
    test_dataset_path: str | None = None
    max_rounds: int = 1
    format_coef: float = 0.0

    async def __call__(self) -> tuple[LeanRLDataset, LeanRLDataset]:

        
        if self.convo_prefix == "standard":
            convo_prefix = LeanEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        
        if not self.train_dataset_path:
            raise ValueError("train_dataset_path is required")
        # if not self.test_dataset_path:
        #     raise ValueError("test_dataset_path is required")
        
        train_ds = LeanRLDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            dataset_path=self.train_dataset_path,
            max_rounds=self.max_rounds,
            format_coef=self.format_coef,
        )
        if self.test_dataset_path:
            test_ds = LeanRLDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                dataset_path=self.test_dataset_path,
                max_rounds=self.max_rounds,
                format_coef=self.format_coef,
            )
        else:
            test_ds = None
            
        return (train_ds, test_ds)


# Add dataset builder mapping
DATASET_BUILDER_MAP = {
    "leanrl": LeanRLDatasetBuilder,
}

def get_lean_dataset_builder(
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    dataset_path: str | None = None,
    max_rounds: int = 1,
    format_coef: float = 0.0,
) -> RLDatasetBuilder:
    """Unified function to get Lean dataset builder"""
    return LeanRLDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        train_dataset_path=dataset_path,
        max_rounds=max_rounds,
        format_coef=format_coef,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the compiler endpoint
    test_code = """
import Mathlib

theorem add_zero (n : ℕ) : n + 0 = n := by
  rfl
"""
    
    print("Testing Lean compiler endpoint...")
    result = compile_lean_1([test_code])
    print(f"Compilation result: {result}")
    
    # Test the environment
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    
    tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
    renderer = renderers.get_renderer("llama3", tokenizer=tokenizer)
    
    env = LeanEnv(
        problem="Prove: n + 0 = n",
        renderer=renderer
    )
    
    # Test format checking
    good_response = "Here's the proof:\n\n```lean4\nimport Mathlib\n\ntheorem add_zero (n : ℕ) : n + 0 = n := by\n  rfl\n```"
    bad_response = "I don't know how to solve this."
    
    print(f"Good response format check: {env.check_format(good_response)}")
    print(f"Bad response format check: {env.check_format(bad_response)}")
