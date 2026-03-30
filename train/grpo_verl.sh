#!/bin/bash

# This script assumes you are running inside a verl environment / container.
# Official repo: https://github.com/volcengine/verl

MODEL_PATH="/mnt/cfs/chubaofs_ads_train_image/ouchuang/bias/0.6B_grc_sft"
TRAIN_RL_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_verl_train.parquet"
VAL_RL_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_verl_val.parquet"
METADATA_CACHE_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_grc/grc_metadata_cache.jsonl"

PROJECT_NAME="grc_verl"
EXPERIMENT_NAME="grc_leaf_brand_grpo"

NUM_GPUS=8
NNODES=1
TOTAL_EPOCHS=5
SAVE_FREQ=20
TEST_FREQ=5

TRAIN_BATCH_SIZE=256
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=256
ROLLOUT_N=4
ROLLOUT_GPU_MEMORY_UTILIZATION=0.6

ACTOR_LR=1e-6
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE_PER_GPU=8
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=8
KL_LOSS_COEF=0.001

GRC_BETA_COR=2.2
GRC_BETA_LAST=2.0
GRC_BETA_LOC=1.0
GRC_BETA_SEM=0.8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

REWARD_FUNC_PATH="${REPO_ROOT}/train/grc_verl_reward.py"
LOCAL_LOG_DIR="./train/logs"
LOG_PATH="${LOCAL_LOG_DIR}/grpo_verl.log"
mkdir -p "${LOCAL_LOG_DIR}"

export GRC_METADATA_PATH="${METADATA_CACHE_PATH}"
export GRC_BETA_COR="${GRC_BETA_COR}"
export GRC_BETA_LAST="${GRC_BETA_LAST}"
export GRC_BETA_LOC="${GRC_BETA_LOC}"
export GRC_BETA_SEM="${GRC_BETA_SEM}"

VERL_CMD=(
  python3
  -m
  verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_RL_DATA}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation='error'
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
  algorithm.use_kl_in_reward=False
  custom_reward_function.path="${REWARD_FUNC_PATH}"
  custom_reward_function.name=compute_score
  trainer.critic_warmup=0
  trainer.logger='["console"]'
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${NUM_GPUS}"
  trainer.nnodes="${NNODES}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
)

if [ -n "${VAL_RL_DATA}" ]; then
  VERL_CMD+=(data.val_files="${VAL_RL_DATA}")
fi

echo "Writing GRPO log to ${LOG_PATH}"
"${VERL_CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"
