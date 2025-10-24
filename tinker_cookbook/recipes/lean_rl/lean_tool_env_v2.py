"""
Lean 4 environment for RL training with tool interaction support.

This environment supports tool calling within <think> tags and uses a remote compiler endpoint 
to verify Lean 4 code correctness.
"""
import json
import re
import uuid
from functools import partial
from typing import Literal, Dict, Any, Optional

import chz
from datasets import load_dataset
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder, Action, StepResult, Observation, StopCondition
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tinker_cookbook.recipes.lean_rl.lean_utils_v2 import DeepSeekCoTHandler as LeanCodeHandler
from tinker_cookbook.recipes.lean_rl.lean_utils_v2 import remove_comments, remove_specific_lines, add_header
from tinker_cookbook.recipes.lean_rl.lean_compiler import compile_lean_1, compile_lean_1_async, compile_lean_2_async

import logging
logger = logging.getLogger(__name__)
import json_repair
from tinker_cookbook.recipes.lean_rl.tool_utils import mcp_system_prompt_1_0, lemma_solving_S1
import copy

class LeanToolEnv(ProblemEnv):
    """
    Environment for Lean 4 theorem proving with tool interaction support.
    
    """
    
    def __init__(
        self,
        init_chat_history, # chat history
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        max_rounds: int = 10, 
        mode: str = "normal",
        enable_thinking: bool = False,
        # Tool-specific coefficients
        tool_format_coef: float = 0.5, 
        tool_answer_coef: float = 1.0, 
        lemma_reward_coef: float = 0.1,
        final_reward_coef: float = 1.0,
    ):
        # Generate unique identifier for this environment instance
        self.uid = str(uuid.uuid4())[:8]  # Use first 8 characters for brevity
        
        assert mode in ["normal", "cold_start_tool_round1"]
        if mode == "cold_start_tool_round1":
            self.max_rounds = 1
            self.lemma_reward_coef = 1.0
            # self.tool_format_coef = 1.0
            print(f"[{self.uid}] Using mode={mode}, override max_rounds={self.max_rounds} and lemma_reward_coef={self.lemma_reward_coef}.")
        
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.max_rounds = max_rounds
        self.init_chat_history = init_chat_history
        self.enable_thinking = enable_thinking
        
        # Tool-specific coefficients
        self.tool_format_coef = tool_format_coef
        self.tool_answer_coef = tool_answer_coef
        self.lemma_reward_coef = lemma_reward_coef
        self.final_reward_coef = final_reward_coef

        # WARNING: 我的理解是dataset管理题目, ENV管理交互逻辑, 因此ENV中做prompting处理, correct me if I'm wrong
        if len(self.init_chat_history) > 1:
            raise ValueError("when init, chat_history should have only one message from user.")
        
        self.handler = LeanCodeHandler()
        self.original_lean4_code_in_input =  self.handler.extrac_code(init_chat_history[0]["content"])
        if self.original_lean4_code_in_input == "None":
            raise ValueError("No Lean 4 code found in the input")
        # self.original_lean4_code_in_input = self.handler.extract_original_lean4_code(init_chat_history[0]["content"])
        
        # Extract original statement name for episode completion check
        self.original_statement_name = self.handler.extract_statement_name(self.original_lean4_code_in_input)
        if self.original_statement_name is None:
            print(f"[{self.uid}] Warning: Could not extract statement name from original code")
        else:
            print(f"[{self.uid}] Original statement name: {self.original_statement_name}")
        
        # Track conversation state
        self.tool_calls_made = 0
        self.successful_tool_calls = 0
        self.step_count = 0
        self.is_next_round_final = False  # Track if next round is the final round
        
        # Lemma pool to store successfully compiled lemmas
        self.lemma_pool = []
        
        # 添加响应类型计数器
        self.response_type_counts = {
            "tool_call": 0,
            "truncated": 0, 
            "im_end": 0
        }

    @property
    def stop_condition(self) -> StopCondition:
        """Get stop sequences including tool call markers."""
        base_stops = self.renderer.get_stop_sequences()
        # Add tool call specific stop sequences
        tool_stops = ["</tool_call>"]
        tool_stop_tokens = []
        for stop_str in tool_stops:
            tokens = self.renderer.tokenizer.encode(stop_str, add_special_tokens=False)
            tool_stop_tokens.extend(tokens)
        return base_stops + tool_stop_tokens
    
    @property
    def system_prompt(self) -> str:
        return mcp_system_prompt_1_0 + f"6. You have a max number of rounds of tool usage: {self.max_rounds}. Which means you should use less than {self.max_rounds - 1} rounds of tool usage to verify your lemmas and then at least once to verify your final proof.\n"
    
    @property
    def post_user_S1_prompt(self) -> str:
        return lemma_solving_S1

    @property
    def next_round_final_prompt(self) -> str:
        return f"You have tried enough rounds of tool usage. Please now attempt to provide the final proof of the original problem: {self.original_statement_name}, using the lemmas you have proved. Please call the lean_compiler tool to verify your proof."

    def _build_right_generation_prompt(self, conversation: list[renderers.Message], 
        add_generation_prompt: bool = True, 
        enable_thinking: bool = False) -> str:
        # # 确保 enable_thinking设置正确, 但是同时obs需要转换为ModelInput
        # tokens = self.renderer.tokenizer.apply_chat_template(
        #     conversation, 
        #     tokenize=True, 
        #     add_generation_prompt=add_generation_prompt, 
        #     enable_thinking=enable_thinking
        # )
        # return types.ModelInput.from_ints(tokens)
        return self.renderer.build_generation_prompt(conversation)

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return initial observation and stop condition."""
        self.step_count = 0  # 重置step计数器
        self.is_next_round_final = False  # 重置final round状态
        self.user_s1_message = copy.deepcopy(self.init_chat_history)
        self.user_s1_message[0]["content"] = self.user_s1_message[0]["content"] + self.post_user_S1_prompt
        self.current_conversation = [{"role": "system", "content": self.system_prompt}] + self.user_s1_message
        
        initial_obs = self._build_right_generation_prompt(self.current_conversation, 
            add_generation_prompt=True, 
            enable_thinking=self.enable_thinking,
        )
        
        print(f"[{self.uid}] [INIT] {initial_obs.length}t (max_rounds: {self.max_rounds})")
        
        return initial_obs, self.stop_condition

    def _is_truncated_response(self, message: renderers.Message) -> bool:
        """Check if the response is truncated (no proper ending)."""
        content = message.get("content", "")
        
        # 检查是否包含完整的结束标记
        has_im_end = "<|im_end|>" in content
        has_tool_end = "</tool_call>" in content
        
        # 如果没有任何结束标记，可能是截断的
        if not (has_im_end or has_tool_end):
            return True
        
        # 如果包含<tool_call>但没有</tool_call>，也是截断的
        if "<tool_call>" in content and not has_tool_end:
            return True
        
        return False

    async def step(self, action: Action) -> StepResult:
        """Process a step in the environment."""
        self.step_count += 1
        
        # Check if episode should end due to max rounds
        max_rounds_reached = self.step_count >= self.max_rounds
        
        # Parse the action to check for statement name match
        message, parse_success = self.renderer.parse_response(action)
        
        # Check if the generated code contains the same statement name as the original problem
        statement_name_match = self._check_statement_name_match(message)
        
        # Episode ends if either max rounds reached OR statement name matches
        self.episode_done = max_rounds_reached or statement_name_match
        
        # Update final round status: if next step would reach max_rounds, mark as final
        self.is_next_round_final = (self.step_count + 1 >= self.max_rounds)
        
        # 计算token数量
        action_tokens = len(action)
        
        # Add message to conversation
        self.current_conversation.append(message)
        
        # Determine episode end reason for logging
        episode_end_reason = ""
        if self.episode_done:
            if max_rounds_reached:
                episode_end_reason = " [MAX_ROUNDS]"
            elif statement_name_match:
                episode_end_reason = " [STATEMENT_MATCH]"
        
        # 检测响应类型并更新计数器
        if self._is_truncated_response(message):
            self.response_type_counts["truncated"] += 1
            print(f"[{self.uid}] [S{self.step_count}] TRUNCATED ({action_tokens}t){episode_end_reason}")
            return await self._handle_truncated_response(message)
        elif self._is_tool_call(message):
            self.response_type_counts["tool_call"] += 1
            tool_call = self._extract_tool_call(message.get("content", ""))
            tool_name = tool_call.get("tool", "unknown") if tool_call else "parse_failed"

            if not self.episode_done:
                print(f"[{self.uid}] [S{self.step_count}] INTERMEDIATE {tool_name} ({action_tokens}t)")
            else:
                print(f"[{self.uid}] [S{self.step_count}] FINAL {tool_name} ({action_tokens}t){episode_end_reason}")
            return await self._handle_tool_call(message)
        else:
            self.response_type_counts["im_end"] += 1
            print(f"[{self.uid}] [S{self.step_count}] FULL_ANSWER ({action_tokens}t){episode_end_reason}")
            return await self._handle_imend_step()

    async def _handle_imend_step(self) -> StepResult:
        """Handle a im_end (non-tool and non-truncated) step."""
        # Create a continuation prompt, add final round prompt if next round is the final one
        content="Please be concise and write the proof in the <tool_call> tags and use the lean_compiler tool to verify your proof."
        if self.is_next_round_final:
            content = content + self.next_round_final_prompt
        self.current_conversation.append(renderers.Message(role="user", content=content))
        
        return StepResult(
            reward=0.0,
            episode_done=self.episode_done,
            next_observation=self._build_right_generation_prompt(self.current_conversation, 
                add_generation_prompt=True, 
                enable_thinking=self.enable_thinking,
            ),
            next_stop_condition=self.stop_condition,
            metrics={
                "response_type": "im_end",  # 添加响应类型
                "im_end_response": True,
                "tool_calls_made": self.tool_calls_made,
                "successful_tool_calls": self.successful_tool_calls,
                "is_next_round_final": self.is_next_round_final,
                # 添加响应类型计数
                "response_type_counts": self.response_type_counts.copy(),
                "total_responses": sum(self.response_type_counts.values())
            }
        )


    def _is_tool_call(self, message: renderers.Message) -> bool:
        """Check if the message contains a tool call."""
        content = message.get("content", "")
        return "<tool_call>" in content and "</tool_call>" in content
    
    def _check_statement_name_match(self, message: renderers.Message) -> bool:
        """Check if the generated code contains the same statement name as the original problem."""
        if self.original_statement_name is None:
            return False
            
        content = message.get("content", "")
        
        # Extract Lean code from the message
        lean_code = self.handler.extrac_code(content)
        if lean_code == "None":
            return False
        
        # Extract statement name from the generated code
        generated_statement_name = self.handler.extract_statement_name(lean_code)
        
        # Check if it matches the original statement name
        if generated_statement_name == self.original_statement_name:
            print(f"[{self.uid}] Statement name match found: {generated_statement_name}")
            return True
        
        return False
    
    def _prepend_lemma_pool(self, code: str) -> str:
        """Prepend all successful lemmas from the lemma pool to the given code."""
        if not self.lemma_pool:
            return code
        
        # Join all lemmas in the pool with newlines
        lemma_code = "\n\n".join(self.lemma_pool)
        
        # Prepend the lemma code to the input code
        return f"{lemma_code}\n\n{code}"
    
    async def _handle_tool_call(self, message: renderers.Message) -> StepResult:
        """Handle a tool call step with format and answer checking."""
        content = message.get("content", "")
        
        # Check format: JSON parsing and tool call structure validation
        format_correct, tool_name, tool_arguments = self._check_tool_format(content)
        
        # Execute tool call and check answer (only if format is correct)
        if format_correct and tool_name and tool_arguments:
            tool_result = await self._call_tool(tool_name, tool_arguments)
            answer_correct = self._check_tool_answer(tool_name, tool_result)
        else:
            # Skip answer checking if format is incorrect
            tool_result = {"status": "error", "message": "Invalid tool call format"}
            answer_correct = False
        
        # Calculate rewards
        format_reward = self.tool_format_coef * format_correct
        answer_reward = self.tool_answer_coef * answer_correct
        total_reward = format_reward + answer_reward
        
        # Update counters
        self.tool_calls_made += 1
        if answer_correct:
            self.successful_tool_calls += 1
        
        # Log results
        status = "✓" if answer_correct else "✗"
        tool_display = tool_name if tool_name else "parse_failed"
        print(f"[{self.uid}]   {status} {tool_display} - Format: {format_correct}, Answer: {answer_correct}, Reward: {total_reward:.2f}")
        
        # Create tool response message
        user_response = self._format_user_tool_response(tool_result)
        self.current_conversation.append(user_response)

        if self.episode_done:
            reward = total_reward*self.final_reward_coef
        else:
            reward = total_reward*self.lemma_reward_coef
        return StepResult(
            reward=reward,
            episode_done=self.episode_done,
            next_observation=self._build_right_generation_prompt(self.current_conversation, 
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            ),
            next_stop_condition=self.stop_condition,
            metrics={
                "response_type": "tool_call",  # 添加响应类型
                "tool_calls_made": self.tool_calls_made,
                "successful_tool_calls": self.successful_tool_calls,
                "tool_format_correct": format_correct,
                "tool_answer_correct": answer_correct,
                "tool_reward": total_reward,
                "is_next_round_final": self.is_next_round_final,
                # 添加响应类型计数
                "response_type_counts": self.response_type_counts.copy(),
                "total_responses": sum(self.response_type_counts.values())
            }
        )

    def _check_tool_format(self, content: str) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Check if tool call has valid format and parse JSON.
        
        Returns:
            (format_correct, tool_name, tool_arguments)
        """
        try:
            # Extract tool call from content
            tool_call = self._extract_tool_call(content)
            if not tool_call:
                return False, None, None
            
            # Basic structure check
            if not isinstance(tool_call, dict):
                return False, None, None
            if "tool" not in tool_call or "arguments" not in tool_call:
                return False, None, None
            
            tool_name = tool_call["tool"]
            tool_arguments = tool_call["arguments"]
            
            # Tool-specific format checks
            if tool_name == "lean_compiler":
                # maybe override tool_arguments
                format_valid, tool_arguments = self._check_lean_compiler_format(tool_arguments)
            elif tool_name == "lean_searcher":
                format_valid = self._check_lean_searcher_format(tool_arguments)
            else:
                format_valid = False  # Unknown tool
            
            if format_valid:
                return True, tool_name, tool_arguments
            else:
                return False, tool_name, tool_arguments
                
        except Exception as e:
            logger.warning(f"Tool format check failed: {e}")
            return False, None, None
    
    def _check_lean_compiler_format(self, arguments: Dict[str, Any]):
        try:
            if "code" not in arguments:
                return False
            
            code = arguments["code"]
            code = remove_comments(code)
            if self.episode_done:
                code = self.handler.problem_check(self.original_lean4_code_in_input, code)
            code = remove_specific_lines(code).strip()
            return True, {"code": code}
        except Exception as e:
            logger.warning(f"Failed to extract lean code: {e}\n content: {code}")
            return False, None
    
    def _check_tool_answer(self, tool_name: str, tool_result: Dict[str, Any]) -> bool:
        """Check if tool execution was successful."""
        if not isinstance(tool_result, dict):
            return False
        
        status = tool_result.get("status", "error")
        return status == "success"

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with given arguments."""
        if tool_name == "lean_compiler":
            return await self._call_lean_compiler(arguments)
        elif tool_name == "lean_searcher":
            return await self._call_lean_searcher(arguments)
        else:
            return {
                "status": "error",
                "message": f"Unknown tool: {tool_name}"
            }
    
    async def _call_lean_compiler(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call the lean compiler tool."""
        code = arguments.get("code", "")
        if not code:
            return {
                "status": "error", 
                "message": "No code provided to lean_compiler"
            }
        
        try:
            # Store original code for potential addition to lemma pool
            original_code = code
            
            # Prepend lemma pool to the code
            code = self._prepend_lemma_pool(code)
            
            code = remove_comments(code)
            # code = remove_specific_lines(code)
            code = add_header(code)
            code = code.strip()
            # print(f"[{self.uid}] [Node A] Calling lean_compiler with code:\n" + "------------------------------" + f"\n{code}" + "\n------------------------------")
            compile_results = await compile_lean_2_async([code])
            if compile_results is None or len(compile_results) == 0:
                return {
                    "status": "error",
                    "message": "Compilation failed - no results returned"
                }
            
            compile_result = compile_results[0]
            if not isinstance(compile_result, dict) or 'compilation_result' not in compile_result:
                return {
                    "status": "error",
                    "message": f"Invalid compile result format: {compile_result}"
                }
            
            comp_result = compile_result['compilation_result']
            is_pass = comp_result.get('pass', False)
            is_complete = comp_result.get('complete', False)
            
            if is_pass and is_complete:
                # Add the original code to lemma pool if it's not already there
                if original_code not in self.lemma_pool:
                    self.lemma_pool.append(original_code)
                    print(f"[{self.uid}] Added lemma to pool (total: {len(self.lemma_pool)})")
                
                return {
                    "status": "success",
                    "message": f"Code is correct, which means the lemma is correct. Please continue to write the next lemma or the final theorem: {self.original_statement_name}."
                }
            else:
                # remove verbose warnings
                comp_result.pop('warnings', None) 
                error_msg = str(comp_result)
                return {
                    "status": "error",
                    "message": f"Code is incorrect, which means the lemma is incorrect. Error message: {error_msg}. Please correct the lemma and try again. Or you can revise the proof plan to prove the original problem: {self.original_statement_name}.",
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Compilation error: {str(e)}"
            }
    
    def _check_tool_answer(self, tool_name: str, tool_result: Dict[str, Any]) -> bool:
        """Check if tool execution was successful."""
        if not isinstance(tool_result, dict):
            return False
        
        status = tool_result.get("status", "error")
        return status == "success"

    async def _handle_truncated_response(self, message: renderers.Message) -> StepResult:
        """Handle a truncated response by asking for continuation."""
        
        # Create a continuation prompt, add final round prompt if next round is the final one
        content="Please continue your response, but keep it concise."
        if self.is_next_round_final:
            content = content + self.next_round_final_prompt
        self.current_conversation.append(renderers.Message(role="user", content=content))
        
        # Return step result to continue the conversation
        return StepResult(
            reward=0.0,
            episode_done=self.episode_done,
            next_observation=self._build_right_generation_prompt(self.current_conversation, 
                add_generation_prompt=True, 
                enable_thinking=self.enable_thinking,
            ),
            next_stop_condition=self.stop_condition,
            metrics={
                "response_type": "truncated",  # 添加响应类型
                "truncated_response": True,
                "tool_calls_made": self.tool_calls_made,
                "successful_tool_calls": self.successful_tool_calls,
                "is_next_round_final": self.is_next_round_final,
                # 添加响应类型计数
                "response_type_counts": self.response_type_counts.copy(),
                "total_responses": sum(self.response_type_counts.values())
            }
        )
    
    def _extract_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from content."""
        try:
            # Find tool call block
            tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
            if not tool_call_match:
                return None
            
            tool_call_json = tool_call_match.group(1)
            tool_call = json_repair.loads(tool_call_json)
            
            # Validate tool call structure
            if "tool" not in tool_call or "arguments" not in tool_call:
                return None
            
            return tool_call
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse tool call: {e}\n content: {content}")
            return None
    
    def _format_user_tool_response(self, tool_result: Dict[str, Any]) -> renderers.Message:
        """Format tool result as a message."""
        content = f"<tool_results>\n{str(tool_result)}\n</tool_results>"
        # WARNING: 将 hint 放在user中, 但是不放在<lean_compiler_result>中
        if self.is_next_round_final:
            content = content + self.next_round_final_prompt
        else:
            content = content + f"Now please remember you have {self.max_rounds - self.step_count} rounds of tool usage left. It's time to get your {self.step_count+1}th step of the proof. "
        return renderers.Message(role="user", content=content)
    
    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        # for now just return empty list
        return []


# Keep existing dataset classes but update to use LeanToolEnv
from datasets import Dataset
class LeanToolRLDataset(RLDataset):
    """
    Dataset for Lean 4 theorem proving problems with tool support.
    """
    
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        dataset_path: str | None = None,
        dataset_type: str = "json",  # "json" or "minif2f"
        max_rounds: int = 10,
        enable_thinking: bool = False,
        mode: str = "normal",
        tool_format_coef: float = 0.5,
        tool_answer_coef: float = 1.0,
        lemma_reward_coef: float = 0.1,
        final_reward_coef: float = 1.0,
    ):
        if dataset_path:
            if dataset_type == "minif2f":
                # Load minif2f dataset using DeepSeekCoTHandler
                try:
                    print(f"📂 Loading minif2f dataset from: {dataset_path}")
                    handler = LeanCodeHandler()
                    split = 'test'  # or 'train' depending on your needs
                    initial_data_list = handler.load_split(dataset_path, split)
                    print(f"📊 Loaded {len(initial_data_list)} items from dataset")
                    
                    # Convert minif2f format to the expected format
                    self.ds = []
                    for i, item_data in enumerate(initial_data_list):
                        if not item_data.get("lean4_code"):
                            print(f"⚠️  Skipping item {i}: no lean4_code found")
                            continue
                        
                        try:
                            # Create messages format expected by the environment
                            # Use prover_inference to generate the prompt
                            prompt_str, messages_for_this = handler.prover_inference(
                                item_data["lean4_code"], renderer.tokenizer
                            )
                            
                            # Create the expected message format
                            messages = [{"role": "user", "content": prompt_str}]
                            
                            # Create dataset row in expected format
                            dataset_row = {
                                "messages": messages,
                                "lean4_code": item_data["lean4_code"],
                                "origin_problem_id": item_data.get("origin_problem_id", item_data.get('problem_id', item_data.get('name'))),
                                "original_data": item_data  # Keep original data for reference
                            }
                            self.ds.append(dataset_row)
                        except Exception as e:
                            print(f"⚠️  Error processing item {i}: {e}")
                            continue
                    
                    print(f"✅ Successfully processed {len(self.ds)} valid items")
                    self.ds = Dataset.from_list(self.ds).shuffle(seed=0)
                except Exception as e:
                    print(f"❌ Error loading minif2f dataset: {e}")
                    print(f"📁 Dataset path: {dataset_path}")
                    print(f"🔍 Please check if the file exists and has valid JSON format")
                    raise
            else:
                # Original JSON dataset loading
                raw_ds = load_dataset('json', data_files=dataset_path)
                raw_ds = raw_ds['train']
                
                # WARNING: 我的理解是dataset管理题目, ENV管理交互逻辑, 因此ENV中做prompting处理, correct me if I'm wrong
                self.ds = []
                for row in raw_ds:
                    if "messages" not in row:
                        print(f"[DATASET] No messages found in dataset row: {row}")
                        raise ValueError(f"No messages found in dataset row: {row}")
                    elif len(row["messages"]) > 1:
                        print(f"[DATASET] More than one message found, skipping the row.")
                        continue
                    self.ds.append(row)
                self.ds = Dataset.from_list(self.ds).shuffle(seed=0)
        else:
            raise ValueError("dataset_path is required")

        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.dataset_type = dataset_type
        self.max_rounds = max_rounds
        self.enable_thinking = enable_thinking
        self.mode = mode
        self.tool_format_coef = tool_format_coef
        self.tool_answer_coef = tool_answer_coef
        self.lemma_reward_coef = lemma_reward_coef
        self.final_reward_coef = final_reward_coef
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
            init_chat_history = x.get("messages", "")
            if not init_chat_history:
                logger.warning(f"No chat history found in dataset row: {x}")
                return None
                
            return ProblemGroupBuilder(
                env_thunk=partial(
                    LeanToolEnv, init_chat_history, self.renderer, 
                    convo_prefix=self.convo_prefix, max_rounds=self.max_rounds, 
                    enable_thinking=self.enable_thinking, mode=self.mode,
                    tool_format_coef=self.tool_format_coef,
                    tool_answer_coef=self.tool_answer_coef,
                    lemma_reward_coef=self.lemma_reward_coef,
                    final_reward_coef=self.final_reward_coef,
                ),
                num_envs=group_size,
                dataset_name="leantoolrl",
            )
        except Exception as e:
            logger.warning(f"Failed to create env group builder: {e}")
            return None


@chz.chz
class LeanToolRLDatasetBuilder(RLDatasetBuilder):
    dataset_cls: type[RLDataset] = LeanToolRLDataset
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    train_dataset_path: str | None = None
    test_dataset_path: str | None = None
    dataset_type: str = "json"  # "json" or "minif2f"
    max_rounds: int = 10 
    mode: str = "normal"
    enable_thinking: bool = False
    tool_format_coef: float = 0.5
    tool_answer_coef: float = 1.0
    lemma_reward_coef: float = 0.1
    final_reward_coef: float = 1.0
    
    def __call__(self) -> tuple[LeanToolRLDataset, LeanToolRLDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = LeanToolEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        if not self.train_dataset_path:
            raise ValueError("train_dataset_path is required")
        
        train_ds = self.dataset_cls(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            dataset_path=self.train_dataset_path,
            dataset_type=self.dataset_type,
            max_rounds=self.max_rounds,
            enable_thinking=self.enable_thinking,
            mode=self.mode,
            tool_format_coef=self.tool_format_coef,
            tool_answer_coef=self.tool_answer_coef,
            lemma_reward_coef=self.lemma_reward_coef,
            final_reward_coef=self.final_reward_coef,
        )
        if self.test_dataset_path:
            test_ds = self.dataset_cls(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                dataset_path=self.test_dataset_path,
                dataset_type=self.dataset_type,
                max_rounds=self.max_rounds,
                enable_thinking=self.enable_thinking,
                mode=self.mode,
                tool_format_coef=self.tool_format_coef,
                tool_answer_coef=self.tool_answer_coef,
                lemma_reward_coef=self.lemma_reward_coef,
                final_reward_coef=self.final_reward_coef,
            )
        else:
            test_ds = None
            
        return (train_ds, test_ds)


# Add dataset builder mapping
DATASET_BUILDER_MAP = {
    "leantoolrl": LeanToolRLDatasetBuilder,
}


def create_minif2f_dataset_builder(
    train_dataset_path: str,
    test_dataset_path: str | None = None,
    model_name: str = "Qwen/Qwen3-30B-A3B",
    renderer_name: str = "qwen",
    batch_size: int = 1,
    group_size: int = 1,
    max_rounds: int = 10,
    mode: str = "normal",
    enable_thinking: bool = False,
    tool_format_coef: float = 0.5,
    tool_answer_coef: float = 1.0,
    lemma_reward_coef: float = 0.1,
    final_reward_coef: float = 1.0,
) -> LeanToolRLDatasetBuilder:
    """
    创建一个用于 minif2f 数据集的 LeanToolRLDatasetBuilder。
    
    Args:
        train_dataset_path: minif2f 训练数据集路径
        test_dataset_path: minif2f 测试数据集路径（可选）
        model_name: 模型名称
        renderer_name: 渲染器名称
        batch_size: 批次大小
        group_size: 组大小
        max_rounds: 最大轮数
        mode: 模式
        enable_thinking: 是否启用思考
        tool_format_coef: 工具格式系数
        tool_answer_coef: 工具答案系数
        lemma_reward_coef: 引理奖励系数
        final_reward_coef: 最终奖励系数
    
    Returns:
        LeanToolRLDatasetBuilder 实例
    """
    return LeanToolRLDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=group_size,
        convo_prefix="standard",
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        dataset_type="minif2f",
        max_rounds=max_rounds,
        mode=mode,
        enable_thinking=enable_thinking,
        tool_format_coef=tool_format_coef,
        tool_answer_coef=tool_answer_coef,
        lemma_reward_coef=lemma_reward_coef,
        final_reward_coef=final_reward_coef,
    )

