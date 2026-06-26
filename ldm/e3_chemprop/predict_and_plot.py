import pandas as pd
import numpy as np
import subprocess
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_dir', type=str,
                        default='/home3/qinl0008/projects/GLDM/ldm')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home3/qinl0008/projects/GLDM/ldm/e3_chemprop/model')
    parser.add_argument('--output_dir', type=str,
                        default='/home3/qinl0008/projects/GLDM/ldm/e3_chemprop/results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mic_conditions = [2.0, 4.0, 6.0, 8.0]

    # Step 1: 为每个condition准备输入csv并调用chemprop_predict
    all_preds = {}
    for mic in mic_conditions:
        txt_path = Path(args.generated_dir) / f'generated_mic{mic}_n1000.txt'
        csv_path = output_dir / f'input_mic{mic}.csv'
        pred_path = output_dir / f'preds_mic{mic}.csv'

        # 读取smiles
        with open(txt_path) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        print(f'mic={mic}: {len(smiles_list)} 个分子')

        # 写成chemprop输入格式
        pd.DataFrame({'smiles': smiles_list}).to_csv(csv_path, index=False)

        # 调用chemprop_predict
        cmd = [
            'chemprop_predict',
            '--test_path', str(csv_path),
            '--checkpoint_dir', args.checkpoint_dir,
            '--preds_path', str(pred_path),
            '--smiles_columns', 'smiles'
        ]
        print(f'预测 mic={mic}...')
        subprocess.run(cmd, check=True)

        # 读取预测结果
        preds_df = pd.read_csv(pred_path)
        all_preds[mic] = preds_df['neg_log2_mic'].values
        print(f'mic={mic}: 预测neg_log2_mic均值={preds_df["neg_log2_mic"].mean():.3f}')

    # Step 2: 打印汇总统计
    print('\n=== 预测MIC分布汇总 ===')
    print(f'{"condition":<12} {"mean neg_log2_mic":>18} {"mean MIC (μg/mL)":>18} {"median MIC":>12}')
    for mic in mic_conditions:
        vals = all_preds[mic]
        mean_log = np.mean(vals)
        mean_mic = np.power(2, -mean_log)
        median_mic = np.power(2, -np.median(vals))
        print(f'mic={mic:<8} {mean_log:>18.3f} {mean_mic:>18.4f} {median_mic:>12.4f}')

    # Step 3: 画KDE图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    labels = {2.0: 'MIC=0.25 μg/mL (cond=2)',
              4.0: 'MIC=0.063 μg/mL (cond=4)',
              6.0: 'MIC=0.016 μg/mL (cond=6)',
              8.0: 'MIC=0.004 μg/mL (cond=8)'}

    for mic, color in zip(mic_conditions, colors):
        vals = all_preds[mic]
        kde = gaussian_kde(vals, bw_method=0.3)
        x = np.linspace(-6, 20, 300)
        ax.plot(x, kde(x), color=color, linewidth=2, label=labels[mic])
        ax.axvline(np.mean(vals), color=color, linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Predicted -log₂(MIC)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Predicted MIC Distribution by Condition (E3)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plot_path = output_dir / 'kde_mic_conditions.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f'\nKDE图已保存: {plot_path}')

if __name__ == '__main__':
    main()
