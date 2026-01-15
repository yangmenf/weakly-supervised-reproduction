#!/usr/bin/env python
import sys
import os
sys.path.append('.')

from tools.infer_seg_voc import crf_proc, parser
import logging
from utils.pyutils import setup_logger

# Parse arguments
args = parser.parse_args([
    '--model_path', '/root/autodl-tmp/experiments/MoRe/outputs/2026-01/voc_reproduce_voc_more_13-11-05-12/checkpoints/model_iter_20000.pth',
    '--data_folder', '/root/autodl-tmp/data/VOC/VOCdevkit/VOC2012/',
    '--infer_set', 'val'
])

# Set up paths
base_dir = args.model_path.split('checkpoints/')[0] + f'/{args.infer_set}/'
cpt_name = args.model_path.split('checkpoints/')[-1].replace('.pth','')
args.logits_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/logits")
args.segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/seg_preds")
args.segs_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/seg_preds_rgb")
args.list_folder = 'datasets/voc'

os.makedirs(args.segs_dir, exist_ok=True)
os.makedirs(args.segs_rgb_dir, exist_ok=True)

# Setup logger
setup_logger(filename=os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/crf_results.log"))

print("Starting CRF post-processing...")
print(f"Logits dir: {args.logits_dir}")
print(f"Segs dir: {args.segs_dir}")

# Run CRF processing
crf_score = crf_proc(args=args)

if crf_score is not None:
    print("CRF post-processing completed successfully!")
else:
    print("CRF post-processing failed!")
