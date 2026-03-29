#!/bin/bash

# This script assumes you are running inside a verl environment / container.
# Official repo: https://github.com/volcengine/verl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/path/to/grc_sft_output}"
TRAIN_RL_DATA="${TRAIN_RL_DATA:-./train/grc_verl_train.parquet}"
VAL_RL_DATA="${VAL_RL_DATA:-./train/grc_verl_val.parquet}"
METADATA_CACHE_PATH="${METADATA_CACHE_PATH:-./train/grc_metadata_cache.jsonl}"
REWARD_FUNC_PATH="${REWARD_FUNC_PATH:-${REPO_ROOT}/train/grc_verl_reward.py}"

PROJECT_NAME="${PROJECT_NAME:-grc_verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grc_leaf_brand_grpo}"

export GRC_METADATA_PATH="${METADATA_CACHE_PATH}"
export GRC_BETA_COR="${GRC_BETA_COR:-2.2}"
export GRC_BETA_LAST="${GRC_BETA_LAST:-2.0}"
export GRC_BETA_LOC="${GRC_BETA_LOC:-1.0}"
export GRC_BETA_SEM="${GRC_BETA_SEM:-0.8}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_RL_DATA}" \
  data.val_files="${VAL_RL_DATA}" \
  data.train_batch_size=256 \
  data.max_prompt_length=2048 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="${REWARD_FUNC_PATH}" \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.total_epochs=5
