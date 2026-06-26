import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                        default='/home3/qinl0008/projects/antibiotics/data/chembl/ecoli_mic_clean.csv')
    parser.add_argument('--output_dir', type=str, 
                        default='/home3/qinl0008/projects/GLDM/ldm/e3_chemprop/data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"原始数据: {len(df)} 行")

    df = df[df['mic_ug_ml'] > 0].copy()
    print(f"过滤mic=0后: {len(df)} 行")

    df['neg_log2_mic'] = -np.log2(df['mic_ug_ml'])
    print(f"neg_log2_mic 范围: {df['neg_log2_mic'].min():.3f} ~ {df['neg_log2_mic'].max():.3f}")

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train+n_val]
    test_df  = df.iloc[n_train+n_val:]

    for split, data in [('train', train_df), ('val', val_df), ('test', test_df)]:
        out = data[['canonical_smiles', 'neg_log2_mic']].copy()
        out.columns = ['smiles', 'neg_log2_mic']
        path = output_dir / f'{split}.csv'
        out.to_csv(path, index=False)
        print(f"{split}: {len(out)} 行 -> {path}")

if __name__ == '__main__':
    main()
