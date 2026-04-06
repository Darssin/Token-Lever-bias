#!/usr/bin/env bash
# Usage: ./eval_grc_parallel.sh [EXTRA_ARGS]

echo "Starting 8-GPU Parallel GRC Evaluation..."

MODEL_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/0.6B_grc_sft"
TEST_PARQUET="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_sid_only_data_test.parquet"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
rm -rf ./result
LOG_DIR="./result"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"
echo "Started at: $(date)"

TOTAL_SAMPLES=22363
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / 8))
BATCH_SIZE=64
NUM_BEAMS=20
DRAFT_MAX_TOKENS=6
REFLECTION_MAX_TOKENS=4
CORRECTION_MAX_TOKENS=6

echo "8-GPU Parallel Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Test parquet: $TEST_PARQUET"
echo "  Total samples: $TOTAL_SAMPLES"
echo "  Samples per GPU: $SAMPLES_PER_GPU"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Correction beam search: $NUM_BEAMS"
echo "  Draft max tokens: $DRAFT_MAX_TOKENS"
echo "  Reflection max tokens: $REFLECTION_MAX_TOKENS"
echo "  Correction max tokens: $CORRECTION_MAX_TOKENS"
echo "  Print generations: ENABLED"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

pids=()
for gpu_id in {0..7}; do
    offset=$((gpu_id * SAMPLES_PER_GPU))
    log_file="${LOG_DIR}/gpu_${gpu_id}.log"

    echo "Starting GPU $gpu_id: samples $offset-$((offset + SAMPLES_PER_GPU - 1))"

    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 -u ./eval/test_grc_hitrate.py \
        --model_path "${MODEL_PATH}" \
        --test_parquet_file "${TEST_PARQUET}" \
        --test_batch_size ${BATCH_SIZE} \
        --num_beams ${NUM_BEAMS} \
        --metrics "hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20" \
        --draft_max_new_tokens ${DRAFT_MAX_TOKENS} \
        --reflection_max_new_tokens ${REFLECTION_MAX_TOKENS} \
        --correction_max_new_tokens ${CORRECTION_MAX_TOKENS} \
        --print_generations \
        --sample_num ${SAMPLES_PER_GPU} \
        --sample_offset ${offset} \
        --gpu_id ${gpu_id} \
        --log_file "$log_file" \
        "$@" > "$log_file" 2>&1 &

    pids+=($!)
    sleep 2
done

echo ""
echo "All 8 processes started:"
for i in {0..7}; do
    echo "  GPU $i: PID ${pids[$i]} -> ${LOG_DIR}/gpu_${i}.log"
done

echo ""
echo "Monitor commands:"
echo "  nvidia-smi"
echo "  tail -f ${LOG_DIR}/gpu_0.log"
echo "  tail -f ${LOG_DIR}/gpu_*.log"
echo "  ps aux | grep test_grc_hitrate"

echo ""
echo "Waiting for all processes to complete..."
overall_status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        overall_status=1
    fi
done

python3 -c "
import re
import os

log_dir = '${LOG_DIR}'
metrics = ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
total_metrics = {m: 0.0 for m in metrics}
total_samples = 0

summary_log = f'{log_dir}/summary_results.log'
with open(summary_log, 'w', encoding='utf-8') as f:
    f.write('8-GPU Parallel GRC Evaluation Summary\\n')
    f.write('=' * 60 + '\\n')
    f.write(f'Timestamp: ${TS}\\n')
    f.write(f'Log directory: {log_dir}\\n\\n')

    found_gpus = 0
    for gpu_id in range(8):
        log_file = f'{log_dir}/gpu_{gpu_id}.log'
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as gpu_f:
                content = gpu_f.read()

            gpu_metrics = {}
            for metric in metrics:
                match = re.search(rf'{metric}:\\s+([\\d\\.]+)', content)
                if match:
                    value = float(match.group(1))
                    gpu_metrics[metric] = value

            sample_match = re.search(r'Total samples:\\s*(\\d+)', content)
            if sample_match and gpu_metrics:
                samples = int(sample_match.group(1))
                total_samples += samples
                found_gpus += 1
                f.write(f'GPU {gpu_id}: {samples} samples\\n')
                for metric, value in gpu_metrics.items():
                    total_metrics[metric] += value
                    f.write(f'    {metric}: {value:.4f}\\n')

                for label in ['Top1 draft hit', 'Top1 corrected hit', 'Top1 corrected gain over draft']:
                    stat_match = re.search(rf'{label}:\\s+([\\d\\.\\-]+)', content)
                    if stat_match:
                        f.write(f'    {label}: {float(stat_match.group(1)):.4f}\\n')
            else:
                f.write(f'GPU {gpu_id}: No results found\\n')
        else:
            f.write(f'GPU {gpu_id}: Log file not found\\n')

    f.write('\\n')
    if found_gpus > 0:
        avg_metrics = {m: total_metrics[m] / found_gpus for m in metrics}

        f.write('FINAL AVERAGED RESULTS:\\n')
        f.write('=' * 60 + '\\n')
        for metric, value in avg_metrics.items():
            f.write(f'{metric:>10}: {value:.4f}\\n')
        f.write('=' * 60 + '\\n')
        f.write(f'Total samples: {total_samples}\\n')
        f.write(f'Completed GPUs: {found_gpus}/8\\n')
        f.write('Evaluation completed successfully!\\n')
    else:
        f.write('No valid results found!\\n')

print(f'Summary saved to: {summary_log}')
" | tee -a "${LOG_DIR}/summary_results.log"

exit "${overall_status}"
