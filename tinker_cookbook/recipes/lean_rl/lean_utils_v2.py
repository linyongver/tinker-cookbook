import re
import pandas as pd
import numpy as np
import json
from jload import jload, jsave
import os
import re

def get_error_str(code, errors, error_thres):
    err_str = ""
    code_lines = code.split('\n')
    token_lengths = [len(line) + 1 for line in code_lines]
    
    # error_thres = False

    error_num_thres = 8 if error_thres else error_num_thres

    for i, error in enumerate(errors[:error_num_thres]):
        start_line = error['pos']['line'] - 1
        start_col = error['pos']['column']

        if error['endPos'] is None:
            end_line = start_line
            end_col = len(code_lines[start_line])
        else:
            end_line = error['endPos']['line'] - 1
            end_col = error['endPos']['column']

        start_char_pos = sum(token_lengths[:start_line]) + start_col
        end_char_pos = sum(token_lengths[:end_line]) + end_col
        
        err_str += f"\nError {i + 1}:\n"
        err_str += f"\nCorresponding Code:\n```lean4\n"
        
        error_code = ""
        for ii in range(-4, 0):
            if start_line + ii >= 0:
                error_code += f"{code_lines[start_line + ii]}\n"
        if start_line != end_line:
            error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"
            
            if not error_thres:
                for j in range(start_line + 1, end_line):
                    error_code += f"{code_lines[j]}\n"
            else:
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    leading_spaces = len(code_lines[j]) - len(code_lines[j].lstrip(' '))
                    error_code += "\n" + " " * leading_spaces + "... --[Truncated]-- ...\n"

            error_code += code_lines[end_line][:end_col] + "</error>" + code_lines[end_line][end_col:] + "\n"
        else:
            error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:end_col] + "</error>" + code_lines[start_line][end_col:] + "\n"
        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"
            
        err_str += error_code
        err_str += f"\n```\n"
        err_str += f"\nError Message: {error['data']}\n"
    
    if len(errors) > error_num_thres:
        err_str += f"\n... [Omitted {len(errors) - error_num_thres} more errors] ...\n"
        
    return err_str

def extract_dpsk_instruction(dpsk_str): # dpsk 7b output
    return  dpsk_str.split("<｜User｜>")[1].split("<｜Assistant｜>")[0]

def extract_qwen_instruction(qwen_str): # qwen output
    return  qwen_str.split("<|im_start|>user")[1].split("<|im_end|>")[0].strip()

def load_data_for_correction(base_output_dir_for_prev_round: str, current_correction_round_num: int,
        num_samples_per_problem: int, base_output_template: str):
    print(
        f"Loading data for correction round {current_correction_round_num} from base directory: {base_output_dir_for_prev_round}")

    if current_correction_round_num == 1:
        prev_round_suffix = ""  # R0 files have no suffix
    elif current_correction_round_num > 1:
        prev_round_suffix = f"_corr{current_correction_round_num - 1}"
    else:
        print("Error: load_data_for_correction called with invalid current_correction_round_num (must be >= 1).")
        return []

    prev_inference_file = os.path.join(base_output_dir_for_prev_round, f"to_inference_codes{prev_round_suffix}.json")
    prev_compilation_file = os.path.join(base_output_dir_for_prev_round,
                                         f"code_compilation_repl{prev_round_suffix}.json")

    assert prev_inference_file, f"Error: Required previous inference file not found: {prev_inference_file}"
    assert prev_compilation_file, f"Error: Required previous compilation file not found: {prev_compilation_file}"


    to_inference_data_prev_round = jload(prev_inference_file)
    compilation_results_data_prev_round = jload(prev_compilation_file)

    if base_output_template == "qwen":
        extract_fun = extract_qwen_instruction
    elif base_output_template == "dpsk":
        extract_fun = extract_dpsk_instruction
    else:
        print("unsupported base template")
        raise Exception

    if "messages_history_list"  not in to_inference_data_prev_round[0]:
        for d in to_inference_data_prev_round:
            # print(d["model_input"])
            d["messages_history_list"] = [{"role": "user", "content": extract_fun(d["model_input"])}]


    comp_lookup = {r["name"]: {"result": r["compilation_result"], "code": r["code"]}
                   for r in compilation_results_data_prev_round if
                   isinstance(r, dict) and "name" in r and "compilation_result" in r and "code" in r}

    passed_original_ids = set()
    failed_problem_variants = {}
    for item_prev_round in to_inference_data_prev_round:
        problem_id_variant = item_prev_round.get("problem_id")
        original_problem_id = item_prev_round.get("origin_problem_id")

        if not problem_id_variant or not original_problem_id: continue
        id_maps = item_prev_round.get("id_maps")
        if id_maps is None:
            assert current_correction_round_num == 1, "Only first revision round accepts no id maps input. Please check your input data."
            id_maps = [{"origin_problem_id": original_problem_id}, {"generation_id": problem_id_variant}]
        # if original_problem_id in passed_original_ids: continue

        if problem_id_variant in comp_lookup:
            comp_data = comp_lookup[problem_id_variant]

            if "errors" not in comp_data["result"]:
                continue

            is_pass = comp_data["result"].get("pass", False)
            is_complete = comp_data["result"].get("complete", False)

            if is_pass and is_complete:
                passed_original_ids.add(original_problem_id)
                # if original_problem_id in failed_problem_variants:
                #     del failed_problem_variants[original_problem_id]
            else:
                if original_problem_id not in failed_problem_variants:
                    failed_problem_variants[original_problem_id] = []

                failed_problem_variants[original_problem_id].append({
                    "last_problem_id": problem_id_variant,
                    "origin_problem_id": original_problem_id,
                    "id_maps": id_maps, 
                    "lean4_code": item_prev_round["lean4_code"],
                    "compiled_code_that_failed_in_prev_round": comp_data["code"],
                    "errors_for_compiled_code_from_prev_round": comp_data["result"],
                    "prev_round_llm_raw_output_for_new_prompt": item_prev_round.get("model_output", ""),
                    "history_messages_from_prev_round_for_new_prompt": item_prev_round.get("messages_history_list", [])
                })

    data_for_new_correction_attempts = []
    total_variants = 0
    unique_p = 0
    for original_id, variants in failed_problem_variants.items():
        # if original_id in passed_original_ids:
        #     continue
        unique_p += 1
        total_variants += len(variants)
        for variant_idx, variant_item in enumerate(variants):
            for i in range(num_samples_per_problem):
                new_attempt_item = variant_item.copy()
                problem_id_variant = variant_item["last_problem_id"]
                new_attempt_item["problem_id"] = f"{problem_id_variant}_corr{current_correction_round_num}_g{i}"
                new_attempt_item["id_maps"] = new_attempt_item["id_maps"].copy() + [
                    {F"corr{current_correction_round_num}_id": new_attempt_item["problem_id"]}]
                data_for_new_correction_attempts.append(new_attempt_item)

    print(f"Correction Round {current_correction_round_num}: Identified {unique_p} unique problems with {total_variants} failed variants. " f"Generating {len(data_for_new_correction_attempts)} new samples for LLM inference.")
    return data_for_new_correction_attempts

def remove_comments(text): # remove comments
    # First remove all /- ... -/ blocks
    # text = re.sub(r'/-.*?-/', '', text, flags=re.DOTALL)
    # text = re.sub(r'/- (?!special open -/).*?-/', '', text, flags=re.DOTALL)
    text = re.sub(r'/-{1,2} (?!special open -/).*?-{1,2}/', '', text, flags=re.DOTALL)
    # Then remove -- comments from each line
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Split on -- and keep only the first part
        cleaned_line = line.split('--', 1)[0]
        cleaned_lines.append(cleaned_line)
    # Join back together and remove excessive empty lines
    cleaned_text = '\n'.join(cleaned_lines)
    # Remove multiple consecutive empty lines
    # cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text.strip()

def return_theorem_to_prove(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    pattern = r'((?:theorem).*?:=\s*by\s*sorry)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def return_theorem_to_replace(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    # pattern = r'((?:theorem).*?:=\s*by)'
    pattern = r'((?:^|\s)theorem\s+.*?:=\s*by)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None

def replace_statement_in_proof(statement, proof):
    if ("apply?" in proof) or ("exact?" in proof):
        return F"[[Error]], 'apply?' or 'exact?' is used, which is not allowed."
    stats_re = remove_comments(statement)
    stats_span_= return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in statement.split('\n')])
        return F"[[Error]], can not find 'theorem' and ':= sorry' in {error_app}"
    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in proof.split('\n')])
        return F"[[Error]], can not find 'theorem' and ':=' in {error_app}"
    return stats_re[:stats_span_[1]].replace("sorry", "") + proof_str[span[1]:]

def remove_comments(text): # remove comments
    # First remove all /- ... -/ blocks
    text = re.sub(r'/-.*?-/', '', text, flags=re.DOTALL)
    # Then remove -- comments from each line
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Split on -- and keep only the first part
        cleaned_line = line.split('--', 1)[0]
        cleaned_lines.append(cleaned_line)
    # Join back together and remove excessive empty lines
    cleaned_text = '\n'.join(cleaned_lines)
    # Remove multiple consecutive empty lines
    # cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text.strip()

def add_header(text):
    import_head = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"  
    return import_head + text

def return_theorem_to_prove(text):
    # # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    # pattern = r'((?:theorem).*?:=\s*by\s*sorry)'
    # Pattern that matches from 'theorem' to either ':= by sorry' or ':= by' with any content in between
    pattern = r'((?:theorem).*?:=\s*by(?:\s*sorry)?)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def return_theorem_to_replace(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    # pattern = r'((?:theorem).*?:=\s*by)'
    pattern = r'((?:^|\s)theorem\s+.*?:=\s*by)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None

def replace_statement_in_proof(statement, proof):
    statement_str = return_theorem_to_prove(remove_comments(statement))
    if statement_str is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in statement.split('\n') if x is not None])
        return F"[[Error]], can not find 'theorem' and ':= sorry' in {error_app}"
    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in proof_str.split('\n') if x is not None])
        return F"[[Error]], can not find 'theorem' and ':=' in {error_app}"
    return proof_str[:span[0]] + statement_str.strip("sorry") + proof_str[span[1]:]


class InferenceHandler:
    # Constructor
    def __init__(self):
        pass
    
    def extrac_code(self, inputs):
        pattern = r'```lean4\n(.*?)\n```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return matches[-1]
        pattern = r'```lean4\n(.*?)```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return matches[-1]
        pattern = r'```lean\n(.*?)```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return matches[-1]
        return "None"


    def clean_code_string(self, code_string):
        # Split the code string into lines
        lines = code_string.splitlines()
        
        # Filter out lines that start with specified keywords or are blank
        filtered_lines = [
            line for line in lines 
            if not (line.startswith("import") or line.startswith("set_option") or line.startswith("open") or line.strip() == "")
        ]
        
        # Join the remaining lines back into a single string
        cleaned_code = "\n".join(filtered_lines)
        return cleaned_code

    def extract_statement_name(self, code_string):
        """Extract the statement name (theorem/lemma name) from Lean 4 code."""
        import re
        
        # Pattern to match theorem or lemma declarations
        # Matches: theorem name (params) : type := by ...
        # or: lemma name (params) : type := by ...
        # pattern = r'(?:theorem|lemma)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]'
        pattern = r'(?:theorem|lemma)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:[\(:\{]|\s)'
        
        match = re.search(pattern, code_string)
        if match:
            return match.group(1)
        
        return None

    def prover_inference(self, lean4_code, tokenizer):
        pass  # This method must be implemented by any derived class

    def generate_correction_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,
                                   prev_round_llm_raw_output,
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num):
        # Returns (prompt_str, messages_list_for_this_prompt)
        raise NotImplementedError


    def split_list_into_chunks(self, input_list, num_chunks):
        """Split a list into approximately equal-sized chunks using only Python built-ins."""
        # Make sure input_list is a regular Python list
        input_list = list(input_list)
        
        # Calculate the length of the list
        list_length = len(input_list)
        
        # Calculate the base size for each chunk
        base_chunk_size = list_length // num_chunks
        
        # Calculate how many chunks need an extra element
        # (when the list can't be evenly divided)
        remainder = list_length % num_chunks
        
        chunks = []
        index = 0
        
        # Create each chunk
        for i in range(num_chunks):
            # Determine this chunk's size (add an extra element if needed)
            current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
            
            # If we've reached the end of the list or this chunk would be empty, stop
            if index >= list_length or current_chunk_size == 0:
                break
                
            # Add this chunk to our result
            chunks.append(input_list[index:index + current_chunk_size])
            index += current_chunk_size
        
        return chunks
    
    def load_split(self, input_file, split):
        # data_list = []
        df = pd.read_json(input_file, lines=True)
        # import pdb; pdb.set_trace()
        if split == "none":
            return df.to_dict(orient='records')
        else:
            return df[df.split.apply(lambda x: str(x) == str(split))].to_dict(orient='records')
    
    def problem_check(self,statement, full_code):
        
        return full_code


class DeepSeekCoTHandler(InferenceHandler):
    def __init__(self):
        # print("[using new reward_utils_v2]")
        pass 

    def extrac_code(self, inputs):
        # print("[using new reward_utils_v2]")
        import_head = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"       
        pattern = r'```lean4\n(.*?)\n```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return import_head + matches[-1]
        pattern = r'```lean4\n(.*?)```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return import_head + matches[-1]
        pattern = r'```lean\n(.*?)```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return import_head + matches[-1]
        return "None"

    def prover_inference(self, lean4_code, tokenizer):
        formal_statement = lean4_code.split(":= by")[0] + ":= by sorry" # include sorry https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B
        prompt = F"Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}```\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text, messages
    
    def problem_check(self,statement, full_code):
        full_code = replace_statement_in_proof(statement, full_code)        
        return full_code

    def generate_correction_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,
                                   prev_round_llm_raw_output,
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num):
        original_stmt_for_prompt = lean4_code_original_stmt.split(":= by")[0] + ":= by sorry"

        current_messages = list(history_messages_from_prev_round)

        # Add PREVIOUS assistant's (failed) attempt
        assistant_content = prev_round_llm_raw_output
        current_messages.append({"role": "assistant", "content": assistant_content})

        # Add CURRENT user feedback and request for new attempt
        user_feedback_content = (
            f"The proof (Round {current_correction_round_num - 1}) is not correct. Following is the compilation error message, where we use <error></error> to signal the position of the error.\n\n{error_message_for_prev_round}"
            "\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error message."
        )
        current_messages.append({"role": "user", "content": user_feedback_content})

        prompt_str = tokenizer.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True)
        return prompt_str, current_messages
   
    def extract_original_lean4_code(self, inputs: str) -> str:
        # inputs are the full prompt+model_response things on the fly.
        # First extract content between "user" and "assistant"
        user_assistant_pattern = r"\buser\b\s*([\s\S]*?)\s*\bassistant\b"
        user_content = re.findall(user_assistant_pattern, inputs, re.DOTALL)
        
        if not user_content:
            print(f"[[Error]] No content found between user and assistant tags: input={inputs}")
            return ""
            
        # Then extract Lean4 code from the first user content
        return_code = self.extrac_code(user_content[0])
        if return_code == "None":
            print(f"[[Error]] No Lean 4 code found in the input(reward_utils.py/extract_original_lean4_code): input={inputs}")
            return ""
        else:
            return return_code


class DeepSeekNonCoTHandler(InferenceHandler):
    def __init__(self):
        pass 

    def prover_inference(self, lean4_code, tokenizer):
        formal_statement = lean4_code.split(":= by")[0] + ":= by" # don't include sorry, directly completion
        prompt = F"Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}"
        return prompt, None

    def generate_correction_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,  # Not used by non-chat
                                   prev_round_llm_raw_output,  # Not used by non-chat directly in prompt
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num):
        original_stmt_for_completion = lean4_code_original_stmt.split(":= by")[0] + ":= by"
        commented_errors = '\n'.join(
            [f'-- {line}' for line in error_message_for_prev_round.splitlines() if line.strip()])

        prompt_str = (
            f"-- The previous proof attempt (Round {current_correction_round_num - 1}) resulted in compilation errors:\n"
            f"{commented_errors}\n"
            f"-- Please provide a corrected version. Wrap the proof in ```lean4 and ```."
        )
        return prompt_str, None  # No message list

class KiminaCoTHandler(InferenceHandler):
    def __init__(self):
        pass 

    def prover_inference(self, lean4_code, tokenizer):
        formal_statement = lean4_code.split(":= by")[0] + ":= by"
        # don't include sorry https://huggingface.co/AI-MO/Kimina-Prover-Preview-Distill-7B
        problem = self.clean_code_string(formal_statement)
        prompt = "Think about and solve the following problem step by step in Lean 4."
        prompt += f"\n# Problem:{problem}"""
        prompt += f"\n# Formal statement:\n```lean4\n{formal_statement}\n```\n"

        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text, messages

    def generate_correction_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,
                                   prev_round_llm_raw_output,
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num):
        original_stmt_for_completion = lean4_code_original_stmt.split(":= by")[0] + ":= by"
        cleaned_original_problem_desc = self.clean_code_string(original_stmt_for_completion)
        current_messages = []

        current_messages = list(history_messages_from_prev_round)

        assistant_content = prev_round_llm_raw_output

        current_messages.append({"role": "assistant", "content": assistant_content})

        user_feedback_content = (
            f"The proof (Round {current_correction_round_num - 1}) is not correct. Following is the compilation error message, where we use <error></error> to signal the position of the error.\n\n{error_message_for_prev_round}"
            "\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error message."
        )
        current_messages.append({"role": "user", "content": user_feedback_content})

        prompt_str = tokenizer.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True)
        return prompt_str, current_messages

    def problem_check(self, statement, full_code):
        full_code = replace_statement_in_proof(statement, full_code)        
        return full_code

    def extract_original_lean4_code(self, inputs: str) -> str:
        # inputs are the full prompt+model_response things on the fly.
        # First extract content between "user" and "assistant"
        user_assistant_pattern = r"\buser\b\s*([\s\S]*?)\s*\bassistant\b"
        user_content = re.findall(user_assistant_pattern, inputs, re.DOTALL)
        
        if not user_content:
            print(f"[[Error]] No content found between user and assistant tags: input={inputs}")
            return ""
            
        # Then extract Lean4 code from the first user content
        return_code = self.extrac_code(user_content[0])
        if return_code == "None":
            print(f"[[Error]] No Lean 4 code found in the input(reward_utils.py/extract_original_lean4_code): input={inputs}")
            return ""
        else:
            return return_code


def remove_specific_lines(text):
    lines = text.split('\n')

    filtered_lines = [line for line in lines if not (
            line.strip().startswith('import') or
            line.strip().startswith('set_option') or
            line.strip().startswith('open')
    )]

    return '\n'.join(filtered_lines)




if __name__ == "__main__":
    # from transformers import AutoTokenizer
    # # model_dir = "/scratch/gpfs/CHIJ/yong/trained_models/Qwen2_5-7B-Instruct_dpsk160kcot_plain_1.0e-4_cosine"
    # model_dir = "/home/zy7019/.cache/huggingface/hub/models--AI-MO--Kimina-Prover-Preview-Distill-7B/snapshots/64b15fe8a897293f396f9559efbb436fb39b8d26"
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # messages = [{"role": "system", "content": "You are an expert in mathematics and Lean 4."}, {'content': '111', 'role': 'user'}]
    # text = tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=True,
    #         add_generation_prompt=True
    #     )
    # print(text)




    # kimina
    kimina_example_dict = {
        "input": "system\nYou are an expert in mathematics and Lean 4.\nuser\nThink about and solve the following problem step by step in Lean 4. \n# Formal statement: \n```lean4\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n\n /-If $x$ and $y$ are positive integers such that $xy - 4x + 3y = 215$, what is the minimal possible value of $|x - y|$?-/ \ntheorem lean_workbook_plus_16832  (x y : ℤ)\n  (h₀ : 0 < x ∧ 0 < y)\n  (h₁ : x * y - 4 * x + 3 * y = 215) :\n  15 ≤ |x - y|   := by\n```\n\nassistant\n", 
        "output": "<think>\n# Solving the Integer Programming Problem\n\nI'm going to solve this problem by finding the minimal possible value of $|x - y|$ for positive integers $x$ and $y$ that satisfy the equation $xy - 4x + 3y = 215$.\n\nFirst, let me think about how to approach this problem. We need to:\n\n1. Find all pairs of positive integers $(x, y)$ that satisfy $xy - 4x + 3y = 215$\n2. Compute $|x - y|$ for each pair\n3. Find the minimal value among these\n\nLet me start by trying to solve the equation algebraically to see if I can find a relationship between $x$ and $y$.\n\nThe equation is: $xy - 4x + 3y = 215$\n\nRearranging terms:\n$xy - 4x + 3y = 215$\n$x(y - 4) + 3y = 215$\n$x(y - 4) + 3y - 12 = 203$\n$x(y - 4) + 3(y - 4) = 203$\n$(x + 3)(y - 4) = 203$\n\nAh! Now we have a product of two integers equaling 203. Let's factor 203 to find possible solutions:\n$203 = 7 \\times 29$\n\nSo, we need to find pairs $(a, b)$ such that $a \\times b = 203$ where $a = x + 3$ and $b = y - 4$\nThe factor pairs of 203 are: $(1, 203), (7, 29), (29, 7), (203, 1)$\n\nFor each factor pair $(a, b)$:\n- $a = x + 3 \\implies x = a - 3$\n- $b = y - 4 \\implies y = b + 4$\n\nLet's compute $x$ and $y$ for each factor pair:\n1. $(1, 203): x = 1 - 3 = -2$ (not positive)\n2. $(7, 29): x = 7 - 3 = 4, y = 29 + 4 = 33$\n3. $(29, 7): x = 29 - 3 = 26, y = 7 + 4 = 11$\n4. $(203, 1): x = 203 - 3 = 200, y = 1 + 4 = 5$\n\nSo we have three valid solutions:\n- $(x, y) = (4, 33)$\n- $(x, y) = (26, 11)$\n- $(x, y) = (200, 5)$\n\nNow let's compute $|x - y|$ for each:\n- For $(4, 33)$: $|4 - 33| = 29$\n- For $(26, 11)$: $|26 - 11| = 15$\n- For $(200, 5)$: $|200 - 5| = 195$\n\nThe minimal value is 15, achieved at $(26, 11)$. \n\nLet's verify these solutions in Lean 4. We'll need to use integer programming tactics to solve this kind of equation with constraints.\n\n```tactics\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n```\n\nWe start by importing the necessary libraries and setting up the problem. The `maxHeartbeats 0` option allows more computation time for the tactic solver.\n\n```tactics\ntheorem lean_workbook_plus_16832  (x y : ℤ)\n  (h₀ : 0 < x ∧ 0 < y)\n  (h₁ : x * y - 4 * x + 3 * y = 215) :\n  15 ≤ |x - y|   := by\n```\n\nWe have the theorem statement with:\n- Variables x, y are positive integers\n- They satisfy x * y - 4 * x + 3 * y = 215\n- We need to prove that 15 ≤ |x - y|\n\nTo prove this, we need to:\n1. Find all possible integer solutions (x, y) to the equation with x, y > 0\n2. Compute |x - y| for each\n3. Show that 15 is the smallest\n\nFrom our algebraic manipulation:\n$(x + 3)(y - 4) = 203$\n$203 = 7 \\times 29$\n\nWe found solutions:\n- (4, 33) with |4 - 33| = 29\n- (26, 11) with |26 - 11| = 15 \n- (200, 5) with |200 - 5| = 195\n\nThe minimal value is 15, so we need to show that no smaller value is possible.\n\nLet's complete the proof in Lean. We'll use omega to solve the arithmetic goals:\n\n```tactics\nhave h2 : x ≤ 200 := by\n  nlinarith [h₁, h₀.left, h₀.right]\n```\n\nThis gives an upper bound on x. Let's think about why:\nFrom $xy - 4x + 3y = 215$, we can rearrange to $xy + 3y = 4x + 215$\n$y(x + 3) = 4x + 215$\n$y = \\frac{4x + 215}{x + 3}$\n\nFor $y$ to be positive, $\\frac{4x + 215}{x + 3} > 0$. Since $x > 0$, the numerator is positive, so denominator must be positive, which it is ($x + 3 > 0$ for $x > 0$).\n\nAlso, to find an upper bound for $x$, consider that $y \\geq 1$:\n$\\frac{4x + 215}{x + 3} \\geq 1$\n$4x + 215 \\geq x + 3$\n$3x \\geq -212$\n$x \\geq -\\frac{212}{3}$\n\nBut $x > 0$, so not helpful for an upper bound.\n\nLet's try another approach:\n$xy - 4x + 3y = 215$\n$xy + 3y = 4x + 215$\n$y(x + 3) = 4x + 215$\n$y = \\frac{4x + 215}{x + 3}$\n\nFor $y$ to be an integer, $x + 3$ must divide $4x + 215$.\n\nLet's rewrite $4x + 215 = (x + 3) \\cdot q + r$ where $0 \\leq r < x + 3$\n\nIf $q = 4$, then $r = 215 - 4 \\cdot 3 = 215 - 12 = 203$\nSo $x + 3$ must divide 203, which is $7 \\cdot 29$\n\nThus $x + 3 \\in \\{1, 7, 29, 203\\}$ (since $x + 3 > 0$ as $x > 0$)\n$x + 3 = 1 \\implies x = -2$ (not positive)\n$x + 3 = 7 \\implies x = 4$\n$x + 3 = 29 \\implies x = 26$\n$x + 3 = 203 \\implies x = 200$\n\nThese are our possible x values: 4, 26, 200\n\nFor each x, we get a corresponding y:\n- $x = 4 \\implies y = \\frac{4 \\cdot 4 + 215}{4 + 3} = \\frac{16 + 215}{7} = \\frac{231}{7} = 33$\n- $x = 26 \\implies y = \\frac{4 \\cdot 26 + 215}{26 + 3} = \\frac{104 + 215}{29} = \\frac{319}{29} = 11$\n- $x = 200 \\implies y = \\frac{4 \\cdot 200 + 215}{200 + 3} = \\frac{800 + 215}{203} = \\frac{1015}{203} = 5$\n\nSo indeed, $x \\in \\{4, 26, 200\\}$, and the maximum value is 200.\n\nIn Lean, how can we complete the proof? We need to show that for any solution, $|x - y| \\geq 15$:\n\n```tactics\nhave h3 : y ≤ 33 := by\n  nlinarith [h₁, h₀.left, h₀.right]\n```\n\nSimilarly, let's think about why $y \\leq 33$:\nUsing $y = \\frac{4x + 215}{x + 3}$, and $x \\geq 4$ (from our solutions), we can check:\nFor $x = 4$, $y = 33$\nFor $x = 26$, $y = 11$\nFor $x = 200$, $y = 5$\n\nSo indeed, $y \\leq 33$ for all solutions.\n\nNow, we can use these bounds to check all possible values:\n\n```tactics\nby_cases hx : x ≥ y\n· -- If x ≥ y, then |x - y| = x - y\n  rw [abs_of_nonneg (by omega)]\n  omega\n```\n\nHere, we split into cases: either $x \\geq y$ or $x < y$. In the first case, $|x - y| = x - y$. If we can show $x - y \\geq 15$, we're done.\n\nThe tactic `omega` is powerful enough to solve arithmetic goals with integer constraints. It will:\n1. Take the equation $x * y - 4*x + 3*y = 215$\n2. The constraints $0 < x$ and $0 < y$\n3. The bound $x \\leq 200$ and $y \\leq 33$\n4. The assumption $x \\geq y$\n\nFrom these, it can deduce that $x - y \\geq 15$. Let's see why:\nFrom our solutions:\n- (4, 33): $4 - 33 = -29$\n- (26, 11): $26 - 11 = 15$\n- (200, 5): $200 - 5 = 195$\n\nThe minimal value in the case $x \\geq y$ is 15, achieved at (26, 11). Since (26, 11) is a valid solution, and omega can find this, it will prove $15 \\leq x - y$ when $x \\geq y$.\n\nNow for the second case:\n\n```tactics\n· -- If x < y, then |x - y| = y - x\n  rw [abs_of_neg (by omega)]\n  omega\n```\n\nSimilarly, when $x < y$, we have $|x - y| = y - x$. The tactic `omega` will prove that $y - x \\geq 15$ given our constraints. \n\nFrom our solutions:\n- (4, 33): $33 - 4 = 29$\n- (26, 11): $11 - 26 = -15$\n- (200, 5): $5 - 200 = -195$\n\nWait, actually, we made an error in interpreting $y - x$ for our solutions:\n\nFor (4, 33): $|4 - 33| = 29$\nFor (26, 11): $|26 - 11| = 15$\nFor (200, 5): $|200 - 5| = 195$\n\nThe minimal value is 15, which is $33 - 26 = 7$ in the $y - x$ case? No, actually $33 - 26 = 7$, but when $x < y$, $|x - y| = y - x$.\n\nWait, let's recalculate:\n\n$|4 - 33| = 29$ (can't be written as $y - x$ for some ordering)\n$|26 - 11| = 15$ (can be written as $26 - 11$)\n$|200 - 5| = 195$ (can be written as $200 - 5$)\n\nSo in the case $x < y$, we should have $y - x \\geq 15$\n\nFor (4, 33): $33 - 4 = 29$\nFor (26, 11): $11 - 26 = -15$\nFor (200, 5): $5 - 200 = -195$\n\nBut since $|x - y| = y - x$ when $x < y$, we need to show that $y - x \\geq 15$ in this case. \n\nThe omega tactic is proving that given our equation and constraints, $y - x \\geq 15$ when $x < y$.\n\nSo our complete proof is showing that regardless of whether $x \\geq y$ or $x < y$, we have $|x - y| \\geq 15$.\n\nTherefore, the minimal possible value of $|x - y|$ is 15, achieved at $(26, 11)$.\n\nThe Lean proof is remarkably concise because:\n1. We bounded x and y to be at most 200 and 33 respectively\n2. The omega tactic was powerful enough to solve the arithmetic goals by checking all possible integer values within these bounds that satisfy the equation\n\nThe key insight algebraically was factoring to get $(x + 3)(y - 4) = 203$, then finding the factor pairs of 203 to determine the possible integer solutions.\n\nThe Lean proof is elegant because it leverages Lean's automation to solve a problem that would be tedious to do by hand checking all integer values. The tactic `omega` is particularly powerful for solving arithmetic goals over integers with constraints.\n</think>\n```lean4\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n\n /-If $x$ and $y$ are positive integers such that $xy - 4x + 3y = 215$, what is the minimal possible value of $|x - y|$?-/ \ntheorem lean_workbook_plus_16832  (x y : ℤ)\n  (h₀ : 0 < x ∧ 0 < y)\n  (h₁ : x * y - 4 * x + 3 * y = 215) :\n  15 ≤ |x - y|   := by\n    have h2 : x ≤ 200 := by\n      nlinarith [h₁, h₀.left, h₀.right]\n    have h3 : y ≤ 33 := by\n      nlinarith [h₁, h₀.left, h₀.right]\n    by_cases hx : x ≥ y\n    · -- If x ≥ y, then |x - y| = x - y\n      rw [abs_of_nonneg (by omega)]\n      omega\n    · -- If x < y, then |x - y| = y - x\n      rw [abs_of_neg (by omega)]\n      omega\n```\n", "score": 0.0, "step": 1}
    

    qwen_dpsk_example_dict = {"input": "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\n\nComplete the following Lean 4 code:\n\n```lean4\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n\ntheorem hyperbola_eccentricity_from_asymptote \n    (a b : ℝ) \n    (ha : a > 0)\n    (hb : b > 0)\n    (h_asymptote : b/a = Real.sqrt 2) : \n    let c := Real.sqrt (a^2 + b^2)\n    let e := c/a\n    e = Real.sqrt 3 := by\n  sorry\n```\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.\nassistant\n", "output": "### Detailed Proof and Analysis\n\nFirst, let's understand the problem:\nWe have a hyperbola with semi-major axis `a` and semi-minor axis `b` such that `b/a = √2`. We are to find the eccentricity `e = c / a` where `c = √(a² + b²)`. We need to prove that `e = √3`.\n\n#### Given:\n1. `a > 0` and `b > 0`.\n2. `b / a = √2`.\n\n#### To Prove:\n`c / a = √3`, where `c = √(a² + b²)`.\n\n#### Approach:\n1. Compute `b²` in terms of `a²` using the given condition `b / a = √2`:\n   \\[\n   b = a \\sqrt{2}\n   \\]\n   Squaring both sides:\n   \\[\n   b^2 = a^2 \\cdot 2 = 2a^2\n   \\]\n\n2. Compute `c² = a² + b²`:\n   \\[\n   c^2 = a^2 + b^2 = a^2 + 2a^2 = 3a^2\n   \\]\n   Taking the square root (and noting that `c > 0` because `a > 0` and `b > 0` implies `a² + b² > 0`):\n   \\[\n   c = \\sqrt{3a^2} = a \\sqrt{3}\n   \\]\n\n3. Compute the eccentricity `e = c / a`:\n   \\[\n   e = \\frac{c}{a} = \\frac{a \\sqrt{3}}{a} = \\sqrt{3}\n   \\]\n\n#### Proof:\n1. From `b / a = √2`, we get `b = a √2`, since `a > 0`. Squaring both sides gives `b² = 2a²`.\n2. Substitute `b² = 2a²` into the expression for `c²`:\n   \\[\n   c^2 = a^2 + b^2 = a^2 + 2a^2 = 3a^2\n   \\]\n   Therefore:\n   \\[\n   c = \\sqrt{3a^2} = a \\sqrt{3}\n   \\]\n3. Compute `e`:\n   \\[\n   e = \\frac{c}{a} = \\frac{a \\sqrt{3}}{a} = \\sqrt{3}\n   \\]\n\n### Step 1: Abstract Plan\n\n1. **Derive `b² = 2a²`:**\n   - Square `b / a = √2` to get `b² / a² = 2`, so `b² = 2a²`.\n\n2. **Compute `c²`:**\n   - Substitute `b² = 2a²` into `c² = a² + b²` to get `c² = 3a²`.\n\n3. **Solve for `c`:**\n   - Compute `c = √(3a²) = a \\sqrt{3}` (since `c > 0` and `a > 0`).\n\n4. **Compute `e`:**\n   - Simplify `e = c / a` to `√3`.\n\n### Step 2: Lean 4 `have` Statements\n\n```lean4\ntheorem hyperbola_eccentricity_from_asymptote \n    (a b : ℝ) \n    (ha : a > 0)\n    (hb : b > 0)\n    (h_asymptote : b/a = Real.sqrt 2) : \n    let c := Real.sqrt (a^2 + b^2)\n    let e := c/a\n    e = Real.sqrt 3 := by\n  have h_b_sq : b^2 = 2 * a^2 := by sorry\n  have h_c_sq : a^2 + b^2 = 3 * a^2 := by sorry\n  have h_c : Real.sqrt (a^2 + b^2) = a * Real.sqrt 3 := by sorry\n  have h_main : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 3 := by sorry\n  sorry\n```\n\n### Explanation of `have` statements:\n1. `h_b_sq`: We will prove `b² = 2 * a²` using the given condition.\n2. `h_c_sq`: We will prove `a² + b² = 3 * a²` by substituting `b²` from `h_b_sq`.\n3. `h_c`: We will prove `√(a² + b²) = a * √3` using `h_c_sq` and simplifying the square root.\n4. `h_main`: We will prove `(√(a² + b²)) / a = √3` using `h_c` and by simplifying the fraction.\n\nThese steps directly correspond to the abstract approach and will form a complete proof when filled in.\n\n### Complete Lean 4 Proof\n\n```lean4\ntheorem hyperbola_eccentricity_from_asymptote \n    (a b : ℝ) \n    (ha : a > 0)\n    (hb : b > 0)\n    (h_asymptote : b/a = Real.sqrt 2) : \n    let c := Real.sqrt (a^2 + b^2)\n    let e := c/a\n    e = Real.sqrt 3 := by\n  have h_b_sq : b^2 = 2 * a^2 := by\n    have h₁ : b / a = Real.sqrt 2 := h_asymptote\n    have h₂ : b = a * Real.sqrt 2 := by\n      have h₃ : a ≠ 0 := by linarith\n      field_simp [h₃] at h₁ ⊢\n      nlinarith [Real.sqrt_nonneg 2, Real.sq_sqrt (show 0 ≤ 2 by norm_num)]\n    have h₃ : b ^ 2 = 2 * a ^ 2 := by\n      rw [h₂]\n      nlinarith [Real.sqrt_nonneg 2, Real.sq_sqrt (show 0 ≤ 2 by norm_num),\n        mul_pos ha hb, Real.sqrt_nonneg 2, Real.sq_sqrt (show 0 ≤ 2 by norm_num)]\n    exact h₃\n  \n  have h_c_sq : a^2 + b^2 = 3 * a^2 := by\n    have h₁ : b ^ 2 = 2 * a ^ 2 := h_b_sq\n    nlinarith\n  \n  have h_c : Real.sqrt (a^2 + b^2) = a * Real.sqrt 3 := by\n    have h₁ : a ^ 2 + b ^ 2 = 3 * a ^ 2 := h_c_sq\n    have h₂ : Real.sqrt (a ^ 2 + b ^ 2) = a * Real.sqrt 3 := by\n      rw [h₁]\n      have h₃ : Real.sqrt (3 * a ^ 2) = a * Real.sqrt 3 := by\n        have h₄ : Real.sqrt (3 * a ^ 2) = a * Real.sqrt 3 := by\n          have h₅ : a > 0 := ha\n          have h₆ : Real.sqrt (3 * a ^ 2) = a * Real.sqrt 3 := by\n            rw [Real.sqrt_eq_iff_sq_eq (by positivity) (by positivity)]\n            nlinarith [Real.sqrt_nonneg 3, Real.sq_sqrt (show 0 ≤ 3 by norm_num)]\n          exact h₆\n        exact h₄\n      rw [h₃]\n      <;> ring_nf\n      <;> field_simp [ha.ne', hb.ne'] <;> nlinarith [Real.sqrt_nonneg 3, Real.sq_sqrt (show 0 ≤ 3 by norm_num)]\n    exact h₂\n  \n  have h_main : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 3 := by\n    have h₁ : (Real.sqrt (a ^ 2 + b ^ 2)) / a = Real.sqrt 3 := by\n      have h₂ : Real.sqrt (a ^ 2 + b ^ 2) = a * Real.sqrt 3 := h_c\n      rw [h₂]\n      have h₃ : (a * Real.sqrt 3) / a = Real.sqrt 3 := by\n        have h₄ : a ≠ 0 := by linarith\n        field_simp [h₄]\n        <;> ring_nf\n        <;> nlinarith [Real.sqrt_nonneg 3, Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num)]\n      rw [h₃]\n      <;> ring_nf\n      <;> nlinarith [Real.sqrt_nonneg 3, Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num)]\n    exact h₁\n  \n  dsimp only\n  exact h_main\n```", "score": 0.0, "step": 3}

    

    kimina_model_input_and_output = kimina_example_dict["input"] + kimina_example_dict["output"]
    qwen_dpsk_model_input_and_output = qwen_dpsk_example_dict["input"] + qwen_dpsk_example_dict["output"]





    # handler = KiminaCoTHandler()
    # solution_str = kimina_model_input_and_output
    handler = DeepSeekCoTHandler()
    solution_str = qwen_dpsk_model_input_and_output

    
    full_proof = handler.extrac_code(solution_str)
    print(f"full_proof=\n{full_proof}" + "\n" + "---"*10)
    original_lean4_code_in_input = handler.extract_original_lean4_code(solution_str)
    print(f"original_lean4_code_in_input=\n{original_lean4_code_in_input}" + "\n" + "---"*10)
    full_code = handler.problem_check(original_lean4_code_in_input, full_proof) 
    print(f"full_code=\n{full_code}" + "\n" + "---"*10)
    code_to_compile = remove_specific_lines(full_code)
    print(f"code_to_compile=\n{code_to_compile}" + "\n" + "---"*10)



    import json
    from tqdm import tqdm
    # mass check:
    # Load the rollout data file
    qwen_dpsk_rollouts_path = "/home/zy7019/verl/CHIJ_checkpoints/5_27_3_GRPO_DistillDeepSeek_genlen10240_bs512_minibs32_n8_lr5e-6_epochs1_kl_loss_coef0.0/rollout_data/1.jsonl"
    # kimina_rollouts_path = "/home/zy7019/verl/CHIJ_checkpoints/5_27_1_GRPO_Kimina7B_genlen10240_bs512_minibs32_n8_kl_loss_coef0.0_ppoepo1_valn32_ChatTemplateV0_kiminaformat_adarft_validated_train/rollout_data/1.jsonl"
    rollouts_path = qwen_dpsk_rollouts_path



    handler = DeepSeekCoTHandler()
    bad_cnt = 0
    total_cnt = 0
    with open(rollouts_path, "r") as f:
        lines = f.readlines()
    
    for idx, line in tqdm(enumerate(lines), desc="Processing examples"):
        # if idx == 10:
        #     pass
        example_dict = json.loads(line)
        
        # Run same code as above for each example
        model_input_and_output = example_dict["input"] + example_dict["output"]
        
        handler = DeepSeekCoTHandler()
        solution_str = model_input_and_output
        
        solution_str = remove_comments(solution_str)
        full_proof = handler.extrac_code(solution_str)
        # print(f"full_proof=\n{full_proof}" + "\n" + "---"*10)
        
        original_lean4_code_in_input = handler.extract_original_lean4_code(solution_str)
        # print(f"original_lean4_code_in_input=\n{original_lean4_code_in_input}" + "\n" + "---"*10)
        
        full_code = handler.problem_check(original_lean4_code_in_input, full_proof)
        # print(f"full_code=\n{full_code}" + "\n" + "---"*10)
        
        code_to_compile = remove_specific_lines(full_code).strip()
        # print(f"code_to_compile=\n{code_to_compile}" + "\n" + "---"*10)
        total_cnt +=1
        if len(code_to_compile)<10:
            bad_cnt +=1
    
    print(f"Bad cnt: {bad_cnt}/{total_cnt}")

# # Example multi-line string
#     statement_string = """
# import Mathlib
# import Aesop
# set_option maxHeartbeats 0
# open BigOperators Real Nat Topology Rat

# theorem lean_workbook_plus_34692_negation
#   :¬( ∀
#       (d₁ d₂ d₃ : ℝ)
#       (h₀ : d₁ = 200)
#       (h₁ : d₂ = 220)
#       (h₂ : d₃ = 88),
#     (d₁ + d₃) / (d₁ + d₂) * 100 = 68.57)
#   := by sorry"""

#     proof_string = """
# import xxx
# open xxx
# set option xxx


# lemma test lemma test_lemma_should_not_impact := by
#     exact


# def jjj hhh
#     block

# lemma test_v2 lemma test_lemma_should_not_impact_v23 := by
#     sorry  
   
# theorem to_proof_theorem_hh **this is the wrong condition** := by 
#     exact
# """

#     # Split into blocks
#     print(replace_statement_in_proof(statement_string, proof_string))


