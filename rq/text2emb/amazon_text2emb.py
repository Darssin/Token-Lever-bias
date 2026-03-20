import argparse
import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from accelerate import Accelerator
from rq.text2emb.common import (
    build_item_text_list,
    generate_item_embeddings,
    load_item_feature_map,
    load_qwen_model,
    save_embeddings,
)


def preprocess_text(args):
    print('Process text data: ')
    print('Dataset: ', args.dataset)
    if args.root:
        print("args.root: ", args.root)
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_item_feature_map(item2feature_path)
    item_text_list = build_item_text_list(item2feature, ['title', 'description'])
    return item_text_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Beauty / Sports / Toys')
    parser.add_argument('--root', type=str, default="")
    # parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--plm_name', type=str, default='qwen')
    parser.add_argument('--plm_checkpoint', type=str, default='xxx', help='Qwen model path')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"Running with {accelerator.num_processes} processes.")

    item_text_list = preprocess_text(args)

    plm_tokenizer, plm_model = load_qwen_model(args.plm_checkpoint)
    
    plm_model = plm_model.to(accelerator.device)
    plm_model.eval()

    item_ids, embeddings = generate_item_embeddings(
        item_text_list,
        plm_tokenizer,
        plm_model,
        accelerator,
        max_sent_len=args.max_sent_len,
        word_drop_ratio=args.word_drop_ratio,
    )
    if accelerator.is_main_process:
        print('Final Embeddings shape: ', embeddings.shape)
        file_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.npy")
        save_embeddings(embeddings, file_path)
