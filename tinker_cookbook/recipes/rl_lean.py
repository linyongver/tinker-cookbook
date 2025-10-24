import asyncio
import os
import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from tinker_cookbook.recipes.lean_rl.lean_env import LeanRLDatasetBuilder
# from tinker_cookbook.rl.train_cli import CLIConfig

@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    load_checkpoint_path: str | None = None

    # Environment configuration (removed - not used in train.Config)

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 5
    kl_penalty_coef: float = 0.0
    num_minibatches: int = 1
    kl_discount_factor: float = 0.0
    remove_constant_reward_groups: bool = False

    # Dataset configuration (for lean RL)
    train_dataset_path: str | None = None
    test_dataset_path: str | None = None
    max_rounds: int = 1
    format_coef: float = 0.0

    # Logging configuration
    log_path: str = chz.field(
        default="/tmp/tinker-examples/rl",
        munger=lambda _, s: os.path.expanduser(s),
    )
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Checkpointing
    save_every: int = 20
    eval_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    save_trajectories: bool = True  # Whether to save trajectory data to JSONL files

    # DAPO (Dynamic Adaptive Policy Optimization) configuration
    use_dapo: bool = False
    dapo_min_eligible_groups: int = 8  # Minimum number of eligible groups needed
    dapo_max_collection_steps: int = 10  # Maximum collection steps per batch
    dapo_all_good_threshold: float = 0.8  # Threshold for all_good groups
    dapo_all_bad_threshold: float = 0.2  # Threshold for all_bad groups


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    lora_rank = cli_config.lora_rank
    batch_size = cli_config.groups_per_batch  # 使用 groups_per_batch 而不是 batch_size
    max_tokens = cli_config.max_tokens
    group_size = cli_config.group_size
    learning_rate = cli_config.learning_rate
    log_parent_path = cli_config.log_path
    exp_name = cli_config.wandb_name
    log_path = os.path.join(log_parent_path, exp_name)
    train_dataset_path = cli_config.train_dataset_path
    test_dataset_path = cli_config.test_dataset_path
    load_checkpoint_path = cli_config.load_checkpoint_path
    eval_every = cli_config.eval_every
    save_every = cli_config.save_every

    print(f"train_dataset_path: {train_dataset_path}")
    print(f"test_dataset_path: {test_dataset_path}")
    print(f"max_rounds: {cli_config.max_rounds}")
    print(f"format_coef: {cli_config.format_coef}")
    
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    builder = LeanRLDatasetBuilder(
        batch_size=batch_size,
        group_size=group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        max_rounds=cli_config.max_rounds,
        format_coef=cli_config.format_coef,
    )

    return train.Config(
        model_name=model_name,
        max_tokens=max_tokens,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=learning_rate,
        save_every=save_every,
        lora_rank=lora_rank,
        eval_every=eval_every,
        load_checkpoint_path=load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        # num_minibatches=cli_config.num_minibatches,
        kl_discount_factor=cli_config.kl_discount_factor,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        base_url=cli_config.base_url,
        # DAPO configuration
        # use_dapo=cli_config.use_dapo,
        # dapo_min_eligible_groups=cli_config.dapo_min_eligible_groups,
        # dapo_max_collection_steps=cli_config.dapo_max_collection_steps,
        # dapo_all_good_threshold=cli_config.dapo_all_good_threshold,
        # dapo_all_bad_threshold=cli_config.dapo_all_bad_threshold,
        # save_trajectories=cli_config.save_trajectories,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)  # 添加这行来获取CLI配置
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
