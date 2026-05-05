#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,6,7}"
export FLOWPLANNER_ROOT="${FLOWPLANNER_ROOT:-$(pwd)}"
export TOOLRL_ROOT="${TOOLRL_ROOT:-/path/to/ToolRL}"
export MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-7B-Instruct}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${TOOLRL_ROOT}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-toolrl_lora_grpo_qwen7b_full3gpu}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

cd "${TOOLRL_ROOT}"

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  data.train_files="${FLOWPLANNER_ROOT}/data/processed/toolrl_benchmark_replan/train.parquet" \
  data.val_files="${FLOWPLANNER_ROOT}/data/processed/toolrl_benchmark_replan/dev.parquet" \
  data.prompt_key=prompt \
  data.max_prompt_length=4096 \
  data.max_response_length=192 \
  data.train_batch_size=3 \
  data.val_batch_size=6 \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora.enabled=True \
  actor_rollout_ref.model.lora.r=8 \
  actor_rollout_ref.model.lora.alpha=16 \
  actor_rollout_ref.model.lora.dropout=0.05 \
  actor_rollout_ref.actor.ppo_mini_batch_size=3 \
  actor_rollout_ref.actor.ppo_micro_batch_size=3 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9000 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=3 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=9000 \
  +actor_rollout_ref.rollout.micro_batch_size=1 \
  critic.model.path="${MODEL_PATH}" \
  critic.model.tokenizer_path="${MODEL_PATH}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_reference_policy=False \
  trainer.project_name=re_toolv2_eval \
  trainer.experiment_name=toolrl_lora_grpo_qwen7b_full3gpu \
  trainer.logger='[console]' \
  trainer.n_gpus_per_node=3 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=null \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  +trainer.val_before_train=True \
  trainer.default_local_dir="${FLOWPLANNER_ROOT}/outputs/toolrl_lora_grpo/qwen7b_full3gpu"
