# Lean 4 RL Training

This directory contains code for training Lean 4 theorem proving using reinforcement learning.

## Overview

LeanEnv is a reinforcement learning environment specifically designed for Lean 4 theorem proving, using a remote compiler endpoint to verify the correctness of model-generated Lean code.

## File Structure

- `train.py`: Main training script
- `test_env.py`: Environment functionality test script  
- `test_lean_utils.py`: Lean code processing utilities test script
- `README.md`: This documentation file

## Core Components

### lean_utils.py

This module contains all necessary functions copied from the original `reward_utils_v2.py`:

- **LeanCodeHandler**: Main code processing class
  - `extract_code()`: Extract Lean code from model responses
  - `extract_original_lean4_code()`: Extract original problem code from input
  - `problem_check()`: Combine problem and proof
  - `process_model_response()`: Complete processing pipeline

- **Helper Functions**:
  - `remove_comments()`: Remove comments from Lean code
  - `return_theorem_to_prove()`: Extract theorem to be proved
  - `return_theorem_to_replace()`: Find theorem position to replace
  - `replace_statement_in_proof()`: Replace statements in proof with original statements
  - `remove_specific_lines()`: Remove import, set_option and other statements

## Environment Features

### LeanEnv Class

- **Inherits from ProblemEnv**: Follows tinker-cookbook's RL environment design pattern
- **Remote Compiler Validation**: Uses Lean compiler endpoint on cluster for code validation
- **Format Check**: Validates if model response contains valid Lean 4 code blocks
- **Answer Validation**: Checks if code compiles successfully and completes proof through compiler
- **Complete Code Processing**: Uses same code extraction and processing logic as original project

### Reward Mechanism

- **Format Reward**: Given for responses containing valid Lean code blocks
- **Correctness Reward**: Given for Lean code that successfully compiles and completes proof
- **Total Reward**: Format reward + Correctness reward

## Usage

### 1. Environment Testing

First run test scripts to ensure environment works properly:
