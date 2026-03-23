# Benchmark Usage

这个目录主要用于评估已经训练好的模型在 `SID prediction` 测试集上的命中率指标。

当前包含 3 个脚本：

- `precompute_global_trie.py`
- `test_model_hitrate.py`
- `eval_parallel_8gpu.sh`

## 1. 输入准备

评估前需要准备：

- 一个可直接加载的模型目录
  - 通常传给 `--merged_model_path`
  - 里面至少要有模型权重和 tokenizer
- 一个测试集 parquet
  - 通常是 `training_prediction_sid_data_test.parquet`
  - 至少包含两列：
    - `description`
    - `groundtruth`
- 一个全局 SID trie 文件
  - 用 `precompute_global_trie.py` 预先生成
  - 供受约束解码使用

测试集的单条样例如下：

```json
{
  "user_id": "1",
  "description": "The user has purchased the following items: <|sid_begin|><s_a_3><s_b_19><s_c_8><s_d_44><|sid_end|>; <|sid_begin|><s_a_5><s_b_9><s_c_11><s_d_20><|sid_end|>;",
  "groundtruth": "<|sid_begin|><s_a_7><s_b_1><s_c_6><s_d_13><|sid_end|>"
}
```

## 2. 先生成全局 Trie

`test_model_hitrate.py` 依赖 `--global_trie_file`。如果这个文件不存在，脚本会直接报错，所以通常先执行：

```bash
python precompute_global_trie.py \
  --test_parquet_file /path/to/training_prediction_sid_data_test.parquet \
  --model_path /path/to/merged_model \
  --output_file /path/to/exact_trie.pkl
```

作用：

- 从测试集的 `description` 和 `groundtruth` 中提取所有合法 SID
- 用模型 tokenizer 把 SID 转成 token 序列
- 构建一个“只允许生成测试集中合法 SID”的精确 trie

输出文件是一个 `pickle`，里面主要包含：

- `exact_trie`
- `valid_sids`
- `valid_sid_tokens`
- `max_length`
- `trie_type = "exact"`

## 3. 单进程评估

最常用的评估脚本是 `test_model_hitrate.py`。

示例：

```bash
python test_model_hitrate.py \
  --merged_model_path /path/to/merged_model \
  --test_parquet_file /path/to/training_prediction_sid_data_test.parquet \
  --global_trie_file /path/to/exact_trie.pkl \
  --test_batch_size 1 \
  --num_beams 20 \
  --max_new_tokens 6 \
  --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
  --log_file ./logs/test_model_hitrate.log
```

常用参数：

- `--merged_model_path`
  - 已合并好的模型目录
- `--additional_lora_path`
  - 可选，在 merged model 之上再额外叠一层 LoRA
- `--test_parquet_file`
  - 测试集 parquet 路径
- `--global_trie_file`
  - 预先生成的 trie 文件
- `--test_batch_size`
  - 评估 batch size
- `--num_beams`
  - beam search 的 beam 数
- `--sample_num`
  - 只评估前多少条，`-1` 表示全部
- `--sample_offset`
  - 从第几条开始，常用于多卡切分
- `--metrics`
  - 支持类似 `hit@1,hit@5,hit@10,ndcg@5,ndcg@10`
- `--print_generations`
  - 打印每条样本的候选生成结果和分数
- `--log_file`
  - 日志输出文件

## 4. 脚本的生成逻辑

`test_model_hitrate.py` 的核心评估方式是：

1. 读取 `description` 作为输入 prompt
2. 用 `model.generate()` 做 beam search
3. 开启 `output_scores=True`
4. 传入 `prefix_allowed_tokens_fn`
5. 让生成过程始终受 trie 约束，只能走向合法 SID

也就是说，对每条输入：

- 会返回 `beam` 条候选 SID 序列
- 同时会拿到每条候选的 `sequences_scores`
- 再据此计算 `hit@k` 和 `ndcg@k`

当前脚本里对应的关键行为包括：

- `num_return_sequences = num_beams`
- `output_scores = True`
- `prefix_allowed_tokens_fn = ...`

如果显存不足，脚本会捕获生成阶段的 CUDA OOM，并自动把 `beam` 减 1 后重试，直到成功或降到 1 仍失败为止。

## 5. 多卡并行评估

`eval_parallel_8gpu.sh` 是一个 8 卡并行评估脚本。它的流程是：

1. 先检查 `GLOBAL_TRIE_FILE` 是否存在
2. 如果不存在，先调用 `precompute_global_trie.py`
3. 按样本数把测试集切成 8 份
4. 每张卡启动一个 `test_model_hitrate.py`
5. 等全部进程结束后，汇总每张卡日志，生成 `summary_results.log`

默认启动方式：

```bash
bash eval_parallel_8gpu.sh
```

也支持把额外参数继续透传给 `test_model_hitrate.py`：

```bash
bash eval_parallel_8gpu.sh --filter_items
```

脚本里当前写死了这些默认值，运行前建议先改成自己的路径：

- `MERGED_MODEL_PATH`
- `ADDITIONAL_LORA_PATH`
- `TEST_PARQUET`
- `GLOBAL_TRIE_FILE`
- `TOTAL_SAMPLES`

日志输出位置：

- 每张卡一份日志：`logs_1010/parallel_eval_<timestamp>/gpu_<id>.log`
- 汇总结果：`logs_1010/parallel_eval_<timestamp>/summary_results.log`

## 6. 结果查看

单卡或单进程评估时，最终日志里会出现：

```text
Final Hit Rate Results:
    hit@1: ...
    hit@5: ...
   hit@10: ...
   ndcg@5: ...
  ndcg@10: ...
```

多卡评估时，最终汇总结果在：

```text
logs_1010/parallel_eval_<timestamp>/summary_results.log
```

## 7. 注意事项

- `--global_trie_file` 是必需的；没有它时，`test_model_hitrate.py` 会直接报错。
- `precompute_global_trie.py` 使用的是 `--model_path` 对应 tokenizer，所以它应与评估模型使用相同词表。
- `eval_parallel_8gpu.sh` 里的 `TOTAL_SAMPLES` 目前是手写常量，不会自动从 parquet 读取；如果测试集大小变了，需要同步修改。
- `test_model_hitrate.py` 默认会把模型直接加载到当前可见 GPU 上，因此运行前要先确认 `CUDA_VISIBLE_DEVICES` 设置正确。
