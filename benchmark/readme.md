# bash指令
```sh
cd /media/cfs/wubintao.6/TLB_demo/benchmark
bash eval_parallel_8gpu_sid_only.sh
```

每次评估需要修改`eval_prarllel_8gpu.sh`中的相关路径设置

1. 待测试的模型路径 MERGED_MODEL_PATH: 直接设置即可
2. 测试集的路径 TEST_PARQUET：设置成待测的数据集的路径
3. 样本数量 TOTAL_SAMPLES：需要根据数据的情况来设置
4. BATCH_SIZE ：搞大一点训练的快一点

```sh
# 下列参数可以进行设置
# 待测试的模型路径 MERGED_MODEL_PATH ，测试集的路径 TEST_PARQUET
MERGED_MODEL_PATH="/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/stage2sft/checkpoint-2796"
TEST_PARQUET="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_sid_only_data_test.parquet"
GLOBAL_TRIE_FILE="./exact_trie.pkl"
TOTAL_SAMPLES=22363
BATCH_SIZE=4
```

数据集路径和samples的匹配见下面

```sh
# beauty
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_sid_only_data_test.parquet
TOTAL_SAMPLES=22363

# sports
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Sports_and_Outdoors/processed_datasets/training_sid_only_data_test.parquet
TOTAL_SAMPLES=35598

# CDs
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/CDs_and_Vinyl/processed_datasets/training_sid_only_data_test.parquet
TOTAL_SAMPLES=75258

# Toys 
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Toys_and_Games/processed_datasets/training_sid_only_data_test.parquet
TOTAL_SAMPLES=19412
```

