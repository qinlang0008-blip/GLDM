import sys
import os
sys.path.append(os.path.join('/home3/qinl0008/projects/GLDM/autoencoder'))
from rdkit import Chem, RDConfig
from rdkit.Chem.QED import qed
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True,
                        help='E1输出的SMILES文件，每行一个')
    parser.add_argument('--train_smiles', type=str, required=True,
                        help='训练集SMILES文件，用于计算Novelty')
    parser.add_argument('--output', type=str, default='e2_metrics.txt')
    return parser.parse_args()

def main():
    args = parse_args()

    # 读取生成的SMILES
    with open(args.generated) as f:
        raw = [line.strip() for line in f if line.strip()]

    # Validity
    mols = [Chem.MolFromSmiles(s) for s in raw]
    valid_pairs = [(s, m) for s, m in zip(raw, mols) if m is not None]
    validity = len(valid_pairs) / len(raw)
    valid_smiles = [s for s, m in valid_pairs]
    valid_mols = [m for s, m in valid_pairs]
    print(f'Validity:   {validity:.4f} ({len(valid_pairs)}/{len(raw)})')

    # Uniqueness（在valid里去重）
    canonical = [Chem.MolToSmiles(m) for m in valid_mols]
    unique = list(set(canonical))
    uniqueness = len(unique) / len(canonical) if canonical else 0
    print(f'Uniqueness: {uniqueness:.4f} ({len(unique)}/{len(canonical)})')

    # Novelty（不在训练集里）
    with open(args.train_smiles) as f:
        train_raw = [line.strip() for line in f if line.strip()]
    train_mols = [Chem.MolFromSmiles(s) for s in train_raw]
    train_canonical = set(Chem.MolToSmiles(m) for m in train_mols if m is not None)
    novel = [s for s in unique if s not in train_canonical]
    novelty = len(novel) / len(unique) if unique else 0
    print(f'Novelty:    {novelty:.4f} ({len(novel)}/{len(unique)})')

    # QED
    qed_scores = [qed(m) for m in valid_mols]
    avg_qed = sum(qed_scores) / len(qed_scores)
    print(f'QED (mean): {avg_qed:.4f}')

    # SA score
    sa_scores = [sascorer.calculateScore(m) for m in valid_mols]
    avg_sa = sum(sa_scores) / len(sa_scores)
    pct_synthesizable = len([s for s in sa_scores if s <= 4.5]) / len(sa_scores)
    print(f'SA (mean):  {avg_sa:.4f}')
    print(f'SA<=4.5:    {pct_synthesizable:.4f}')

    # 保存结果
    with open(args.output, 'w') as f:
        f.write(f'Validity:   {validity:.4f} ({len(valid_pairs)}/{len(raw)})\n')
        f.write(f'Uniqueness: {uniqueness:.4f} ({len(unique)}/{len(canonical)})\n')
        f.write(f'Novelty:    {novelty:.4f} ({len(novel)}/{len(unique)})\n')
        f.write(f'QED (mean): {avg_qed:.4f}\n')
        f.write(f'SA (mean):  {avg_sa:.4f}\n')
        f.write(f'SA<=4.5:    {pct_synthesizable:.4f}\n')
    print(f'Saved to {args.output}')

if __name__ == '__main__':
    main()
