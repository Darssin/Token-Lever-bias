## 模型训练

训练分成两个阶段，第一阶段是对齐，参考 `model_training/run_training_stage1_full.sh` 脚本对模型进行对齐；第二阶段是 sft，训练 spm 任务，参考 `run_training_rec.sh`。

扩充词表等已经预先完成，相对应的模型路径位于

```sh
## base模型
/home/ea-ea-ads-purchasechain-1/models/Qwen3-8B

## 关于拓展词表，统一参考beauty的设置，设置4层码本（256*4），其base模型放于
/home/ea-ea-ads-purchasechain-1/wubintao/OneRec-8B-v2/Qwen3-8B-expand
```



### 一阶段：对齐

这里参考 openonerec 修改了对齐脚本，仅改变新增加词表的 embedding 层权重，训练脚本如下，有些丑陋。

```sh
# 启动脚本
cd /media/cfs/wubintao.6/TLB_demo

TENSORBOARD_DIR=/media/cfs/wubintao.6/experiment/tensorboard/LTB-demo/Sports_and_Outdoors/stage1 \
bash model_training/run_training_stage1_full.sh \
	/home/ea-ea-ads-purchasechain-1/models/Qwen3-8B  \
  /home/ea-ea-ads-purchasechain-1/wubintao/OneRec-8B-v2/Qwen3-8B-expand \
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Sports_and_Outdoors/processed_datasets \
  /mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Sports_and_Outdoors/stage1 
```

后续的算法改进可以基于 beauty 数据集的对齐模型展开尝试，训练后的模型路径位于，后续可以迁移到另外一个网盘。

```sh
/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/stage1/checkpoint-2796
```



### 二阶段：sft

这里完全复用了 onerec-think 的开源代码进行模型训练，只是数据集改成了自己的数据集，相关设置需要在脚本中修改

```sh
# 启动脚本
cd /media/cfs/wubintao.6/TLB_demo/model_training

bash run_training_rec.sh
```

需要注意修改下述参数来确保训练正确

MODEL_DIR: warmup 模型路径

TRAIN_DATA： 训练数据路径

VAL_DATA： 验证集数据路径

--output_dir ： 模型的输出路径

```sh
## 参考路径
# beauty
MODEL_DIR="/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/stage1/checkpoint-2796"
TRAIN_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_prediction_sid_data_train.parquet"
VAL_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_prediction_sid_data_valid.parquet"

## 后面的待定...等我先把base跑出来
# sports
MODEL_DIR="/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/stage1/checkpoint-2796"
TRAIN_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_prediction_sid_data_train.parquet"
VAL_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_prediction_sid_data_valid.parquet"

# CDs
MODEL_DIR="/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/stage1/checkpoint-2796"
TRAIN_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_prediction_sid_data_train.parquet"
VAL_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_prediction_sid_data_valid.parquet"

# Toys 
MODEL_DIR="/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/stage1/checkpoint-2796"
TRAIN_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_prediction_sid_data_train.parquet"
VAL_DATA="/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets/training_prediction_sid_data_valid.parquet"
```

