import sys
sys.path.append('../autoencoder/')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import torch
from omegaconf import OmegaConf
from model_utils import get_params
from dataset import MolerDataset
from moler_ldm import LatentDiffusion
from DDIM import MolSampler
from rdkit import Chem

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0',
                        help='CUDA device(s) to use, e.g. 0 or 0,1')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/ldm_con_mic+wae_con.yml')
    parser.add_argument('--raw_trace_dir', type=str, required=True)
    parser.add_argument('--output_pyg_dir', type=str, required=True)
    parser.add_argument('--mic_value', type=float, default=4.0,
                        help='-log2(MIC), e.g. 4.0 = MIC 0.0625 ug/mL')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--ddim_steps', type=int, default=200)
    parser.add_argument('--ddim_eta', type=float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='CFG guidance scale. 1.0 = no guidance. Requires model trained with cfg_drop_prob > 0.')
    parser.add_argument('--output', type=str, default='generated_smiles.txt')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    config = OmegaConf.load(args.config)

    # 只用valid_0做模型初始化，不加载全部数据
    dataset = MolerDataset(
        root=args.output_pyg_dir,
        raw_moler_trace_dataset_parent_folder=args.raw_trace_dir,
        output_pyg_trace_dataset_parent_folder=args.output_pyg_dir,
        split='valid_0',
        gen_step_drop_probability=0.0,
    )
    print(f'Dataset loaded: {len(dataset)} chunks')

    first_stage_params = get_params(dataset)
    first_stage_params['full_graph_encoder']['layer_type'] = 'FiLMConv'
    first_stage_params['partial_graph_encoder']['layer_type'] = 'FiLMConv'
    # 必须和训练时一致，否则checkpoint load失败
    first_stage_params['gene_exp_condition_mlp']['input_feature_dim'] = 832 + 978 + 1

    first_stage_config = config['model']['first_stage_config']
    ldm_params = config['model']['params']

    ldm_model = LatentDiffusion(
        first_stage_config,
        config['model']['cond_stage_config'],
        dataset,
        drop_prob=0.0,
        batch_size=1,
        first_stage_params=first_stage_params,
        first_stage_ckpt=first_stage_config['ckpt_path'],
        unet_config=config['model']['unet_config'],
        **ldm_params
    )

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    ldm_model.load_state_dict(checkpoint['state_dict'])
    ldm_model.to(device)
    ldm_model.eval()
    print('Model loaded.')

    # condition tensor: encode scalar MIC value through mic_encoder → [B, 1, 64]
    mic_tensor = torch.full((args.n_samples, 1, 1), args.mic_value, device=device)
    conditioning = ldm_model.mic_encoder(mic_tensor)  # [B, 1, 1] → [B, 1, 64]
    print(f'Sampling {args.n_samples} molecules at mic_value={args.mic_value}...')

    # CFG: null condition is the all-zeros embedding (matches training dropout)
    unconditional_conditioning = None
    if args.guidance_scale != 1.0:
        unconditional_conditioning = torch.zeros_like(conditioning)  # [B, 1, 64]
        print(f'Using CFG with guidance_scale={args.guidance_scale}')

    sampler = MolSampler(ldm_model)
    samples, _ = sampler.sample(
        S=args.ddim_steps,
        batch_size=args.n_samples,
        conditioning=conditioning,
        shape=(1, 512),
        ddim_eta=args.ddim_eta,
        unconditional_guidance_scale=args.guidance_scale,
        unconditional_conditioning=unconditional_conditioning,
    )

    z = samples.view((args.n_samples, 512))
    print('Decoding latents...')
    decoder_states = ldm_model.first_stage_model.decode(
        latent_representations=z, max_num_steps=120
    )

    smiles_list = []
    for state in decoder_states:
        mol = state.molecule
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            smiles_list.append(smi)
        else:
            smiles_list.append('INVALID')

    valid = [s for s in smiles_list if s != 'INVALID']
    print(f'Valid: {len(valid)}/{args.n_samples}')

    with open(args.output, 'w') as f:
        for smi in smiles_list:
            f.write(smi + '\n')
    print(f'Saved to {args.output}')

if __name__ == '__main__':
    main()
