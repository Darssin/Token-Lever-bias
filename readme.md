

## TLB



```sh
# 同步脚本
bash ./sync.sh

git clone -b v4.57.1 --depth 1 https://github.com/huggingface/transformers.git
```



```sh
# 数据路径
# 母文件夹
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty

# 核心用到的文件地址
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty

# 中间文件地址
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/pretrain_json_outputs

## 所有模型的路径
/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo

# Beauty
/mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty


```



### 训练数据集相关

#### step1：将数据进行筛选和清洗

```sh
# 数据构造
# amazon数据集，对齐onerec-think的预处理方案

# 解压原始数据集
gunzip指令用于解压缩文件

# 进入目录
cd /media/cfs/wubintao.6/TLB_demo/data

# 依次处理各个原始数据集
python amazon14_data_process.py \
  --meta_file /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Toys_and_Games/meta_Toys_and_Games.json \
  --review_file /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Toys_and_Games/reviews_Toys_and_Games_5.json \
  --output_dir /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Toys_and_Games/processed_Toys

# Beauty 数据集处理好了，与onerec-think对齐,至少数量上对应完全
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty

# Sports_and_Outdoors
Loading reviews...
Loaded raw reviews: 296337
Loading metadata...
Loaded meta items: 532197
Applying iterative k-core filtering...
After filtering: users=35598, items=18357, reviews=296337

/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Sports_and_Outdoors/processed_Sports

# CDs_and_Vinyl数据集
Loading reviews...
Loaded raw reviews: 1097592
Loading metadata...
Loaded meta items: 492799
Applying iterative k-core filtering...
After filtering: users=75258, items=64443, reviews=1097592

/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/CDs_and_Vinyl/processed_CDs

# Toys_and_Games数据集
Loading reviews...
Loaded raw reviews: 167597
Loading metadata...
Loaded meta items: 336072
Applying iterative k-core filtering...
After filtering: users=19412, items=11924, reviews=167597


/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Toys_and_Games/processed_Toys
```

```sh
# 示例的数据集
"11924": {
    "title": "Littlest Pet Shop Blythe and Pet - Buckles & Bows",
    "description": "Get ready for a mini shopping spree with this fun Buckles and Bows set! Your pretty Blythe figure and her cute-as-a-button mouse pet just love to go shopping for accessories and clothes and they love it even better when you pick out their purchases with them! Set your Blythe figure on her doll stand and brush her hair with her comb accessory so she will look great for the shopping trip. She and her mouse pet have to look extra fabulous so you can pick out just the right fashionable styles to go with their looks! Figure and pet come with doll stand, comb and other accessories. Blythe figure B2 with pet 1618. Ages 4 and up.",
    "categories": "Toys & Games > Dolls & Accessories > Playsets"
  }
```



#### Step2：将商品的明文转化为embedding向量

这里我们参考minionerec的方式构造商品的embedding向量

```sh
### 一定要在母目录下面进行

## 得到embedding
python pretrain_json_pipeline/generate_embeddings.py \
  --input_json /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/item_meta.json \
  --plm_checkpoint /home/ea-ea-ads-purchasechain-1/models/Qwen3-Embedding-4B \
  --plm_name qwen3
  
accelerate launch --num_processes 8 pretrain_json_pipeline/generate_embeddings.py \
  --input_json /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/item_meta.json \
  --plm_checkpoint /home/ea-ea-ads-purchasechain-1/models/Qwen3-Embedding-4B \
  --plm_name qwen3
  
# 会在相同路径下得到
pretrain_json_outputs/item_meta.emb-qwen3-td.npy
pretrain_json_outputs/item_meta.item_ids.json

## 进行rq-kmeans
python pretrain_json_pipeline/generate_sids_and_merge.py \
  --input_json /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/item_meta.json \
  --embedding_path /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/pretrain_json_outputs/item_meta.emb-qwen3-td.npy \
  --item_ids_path /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/pretrain_json_outputs/item_meta.item_ids.json \
  --num_levels 4 \
  --codebook_size 256
  
# 得到
Saved index JSON to /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/pretrain_json_outputs/item_meta.index.json
Saved merged JSON to /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/pretrain_json_outputs/item_meta.with_sid.json
Saved item metadata JSON to /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/pretrain_json_outputs/item_meta.item.json

# 这个是有用的
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/pretrain_json_outputs/item_meta.with_sid.json

"12101": {
    "title": "Jessica Mcclintock By Jessica Mcclintock For Women. Eau De Parfum Spray 1.7 Oz.",
    "description": "Introduced in 1987. Fragrance notes: citrus notes of lemon, along with basil, white rose, and white jasmine. Recommended use: daytime.When applying any fragrance please consider that there are several factors which can affect the natural smell of your skin and, in turn, the way a scent smells on you. For instance, your mood, stress level, age, body chemistry, diet, and current medications may all alter the scents you wear. Similarly, factor such as dry or oily skin can even affect the amount of time a fragrance will last after being applied",
    "categories": "Beauty > Fragrance > Women's > Eau de Parfum",
    "sid": "<|sid_begin|><s_a_71><s_b_63><s_c_10><s_d_229><|sid_end|>"
  }
```

#### step3：进行rq-kmeans以及训练评估集的划分

```sh
## 构造几个训练数据集和测试集评估集

bash pretrain_json_pipeline/generate_all_datasets.sh \
  /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/user_sequence.txt \
  /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/item_meta.with_sid.json \
  /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets

# 最终所有数据集都放在下述路径中
/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets

# 存在几个空的数据
grep -A 6 '"13":' /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_beauty/item_meta.with_sid.json

Warning: item_id 8407 for user 18001 missing sid/title/categories, skipping.
Warning: item_id 8407 for user 18001 missing sid/title/categories, skipping.
Warning: item_id 8407 for user 18001 missing sid/title/categories, skipping.
Warning: item_id 13 for user 18051 missing sid/title/categories, skipping.
Warning: item_id 13 for user 18051 missing sid/title/categories, skipping.
Warning: item_id 13 for user 18051 missing sid/title/categories, skipping.
```

### 模型训练相关

```sh
## 关于拓展词表，统一参考beauty的设置，设置4层码本，其base模型放于
/home/ea-ea-ads-purchasechain-1/wubintao/OneRec-8B-v2/Qwen3-8B-expand

## 训练数据集位置
# 内包含training_align_data_test.parquet   training_align_data_valid.parquet          training_prediction_sid_data_train.parquet  training_RA_test.parquet   training_RA_valid.parquet
# training_align_data_train.parquet  training_prediction_sid_data_test.parquet  training_prediction_sid_data_valid.parquet  training_RA_train.parquet

/mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets
```

```sh
## 现在开始训练
TENSORBOARD_DIR=/media/cfs/wubintao.6/TLB_demo/tenasorboard/Beauty/stage1 \
bash model_training/run_training_stage1.sh \

  /home/ea-ea-ads-purchasechain-1/wubintao/OneRec-8B-v2/Qwen3-8B-expand \
  /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets \
  /mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/stage1 
  
TENSORBOARD_DIR=/media/cfs/wubintao.6/experiment/tensorboard/LTB-demo/Beauty/stage1 \
bash model_training/run_training_stage1_full.sh \
	/home/ea-ea-ads-purchasechain-1/models/Qwen3-8B  \
  /home/ea-ea-ads-purchasechain-1/wubintao/OneRec-8B-v2/Qwen3-8B-expand \
  /mnt/cfs/chubaofs_ads_train_image/wubintao/datasets/minionerec/data/amazon_reviews_2014_rpg/Beauty/processed_datasets \
  /mnt/cfs/chubaofs_ads_train_image/wubintao/models/TLB_demo/Beauty/stage1 
```



python pretrain_json_pipeline/generate_sid_only_data.py \
  --user_sequence data/user_sequence.txt \
  --item_meta_with_sid data/item_meta.with_sid.json \
  --output_dir data/output


TENSORBOARD_DIR=../outputs/my_tb \
bash model_training/run_training_sid_only.sh \
  ../basemodel/Qwen3-1-7B-expand \
  ../data \
  ../outputs/sid_only_sft

