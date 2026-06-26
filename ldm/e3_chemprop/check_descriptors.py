import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

generated_dir = Path('/home3/qinl0008/projects/GLDM/ldm')
mic_conditions = [2.0, 4.0, 6.0, 8.0]

print(f'{"condition":<10} {"MW":>8} {"LogP":>8} {"HBD":>6} {"HBA":>6} {"QED":>8} {"Rings":>7}')
print('-' * 60)

for mic in mic_conditions:
    txt_path = generated_dir / f'generated_mic{mic}_n1000.txt'
    with open(txt_path) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    mw, logp, hbd, hba, qed_vals, rings = [], [], [], [], [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mw.append(Descriptors.MolWt(mol))
        logp.append(Descriptors.MolLogP(mol))
        hbd.append(Descriptors.NumHDonors(mol))
        hba.append(Descriptors.NumHAcceptors(mol))
        qed_vals.append(QED.qed(mol))
        rings.append(Descriptors.RingCount(mol))

    print(f'mic={mic:<6} '
          f'{np.mean(mw):>8.1f} '
          f'{np.mean(logp):>8.3f} '
          f'{np.mean(hbd):>6.2f} '
          f'{np.mean(hba):>6.2f} '
          f'{np.mean(qed_vals):>8.3f} '
          f'{np.mean(rings):>7.2f}')
