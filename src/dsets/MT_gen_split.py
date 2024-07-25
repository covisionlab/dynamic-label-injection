from argparse import ArgumentParser

import os
import torch
import random
import numpy as np
import pandas as pd

def ensure_reproducibility(seed: int = 42) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen

METAL_DSET_MAP = {  
    'MT_Free': -1,
    'MT_Blowhole': 0,
    'MT_Break': 1,
    'MT_Crack': 2,
    'MT_Fray': 3,
    'MT_Uneven': 4,
}

def gen_splits(dset_csv, output_dir):
    all_examples = pd.read_csv(dset_csv).sample(frac=1)

    free_idx =  METAL_DSET_MAP['MT_Free']
    
    free_examples = all_examples[all_examples['defect_id'] == free_idx]
    defected_examples = all_examples[all_examples['defect_id'] != free_idx]

    train_data, val_data = [], []
    for defect_id in METAL_DSET_MAP.values():
        if defect_id == free_idx:
            continue
        defect_df = defected_examples[defected_examples['defect_id'] == defect_id] 
        train_n = int(defect_df.shape[0] * 0.8)

        tr = defect_df.sample(n=train_n)
        val = defect_df.drop(tr.index)

        train_data.append(tr)
        val_data.append(val)

    train_defected_data = pd.concat(train_data)
    train_data = pd.concat([train_defected_data, free_examples])
    val_data = pd.concat(val_data)

    train_defected_data.to_csv(output_dir + '/train_defected_data.csv', index=False)
    train_data.to_csv(output_dir + '/train_data.csv', index=False)
    val_data.to_csv(output_dir + '/val_data.csv', index=False)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='seed')
    parser.add_argument('--dset_csv', type=str, required=True, help='path to the dataset csv')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir + f'/{args.seed}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    _ = ensure_reproducibility(args.seed)
    gen_splits(args.dset_csv, output_dir)