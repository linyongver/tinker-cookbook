"""
Lean 4 RL training script.

This script demonstrates how to train a model on Lean 4 theorem proving
using the remote compiler endpoint for reward calculation.
"""

import asyncio
import os

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from tinker_cookbook.recipes.lean_rl.lean_env import LeanDatasetBuilder


def build_config() -> train.Config:
    """
    Build the training configuration for Lean 4 theorem proving.
    """
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    
    # Create the Lean dataset builder
    dataset_builder = LeanDatasetBuilder(
        batch_size=32,  # Number of problem groups per batch
        group_size=8,   # Number of environments per group (for reward centering)
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        convo_prefix="standard",  # Use standard few-shot examples
    )
    
    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/lean-rl",
        dataset_builder=dataset_builder,
        learning_rate=3e-5,  # Lower learning rate for more stable training
        max_tokens=512,      # Allow longer responses for Lean proofs
        eval_every=10,       # Evaluate every 10 batches
        save_every=20,       # Save checkpoint every 20 batches
        lora_rank=32,        # Use LoRA for efficient fine-tuning
    )


def main():
    """
    Main training function.
    """
    config = build_config()
    
    # Check if log directory exists and ask user what to do
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    
    # Create log directory
    os.makedirs(config.log_path, exist_ok=True)
    
    print("Starting Lean 4 RL training...")
    print(f"Model: {config.model_name}")
    print(f"Log path: {config.log_path}")
    print(f"Batch size: {config.dataset_builder.batch_size}")
    print(f"Group size: {config.dataset_builder.group_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max tokens: {config.max_tokens}")
    
    # Run the training
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
