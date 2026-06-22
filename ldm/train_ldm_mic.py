import sys
sys.path.append('../autoencoder/')
import numpy as np
from dataset import MolerDataset
from torch_geometric.loader import DataLoader
import torch
from omegaconf import OmegaConf
from model_utils import get_params
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
from moler_ldm import LatentDiffusion
import argparse

# NOTE: this script is a fork of train_ldm_l1000.py, adapted for the
# antibiotics MIC-conditioned project. Differences from the original:
#   1. LincsDataset (L1000 gene-expression specific) -> MolerDataset (base class).
#      We don't have gene_expressions/dose files; our condition is a single
#      mic_value scalar attached via the `properties` field during preprocessing
#      (see Track B data pipeline docs).
#   2. Default config_file points at config/ldm_con_mic+wae_con.yml
#      (dim/context_dim = 1, cond_stage_key/key = mic_value, and critically
#      first_stage_config.ckpt_path points at first_stage_only.ckpt, a
#      re-packaged checkpoint extracted from GLDM_WAE_cond.pt — see handover
#      notes on why the original ckpt_path is stale and why GLDM_WAE_cond.pt
#      can't be pointed to directly without stripping the 'first_stage_model.'
#      prefix).
#   3. New --raw_trace_dir / --output_pyg_dir args replace the hardcoded L1000
#      data paths.
#   4. The `gene_exp_condition_mlp` input_feature_dim override (832+978+1) is
#      KEPT AS-IS on purpose: this MLP belongs to the frozen first-stage model
#      and is never invoked during LDM training (encode_first_stage() only
#      calls full_graph_encoder/partial_graph_encoder), but its shape must
#      still match what's stored in the checkpoint for load_from_checkpoint
#      to succeed (using_lincs=True in the YAML instantiates it).


def filter_dataset(remove_idx, dataset):
    mask = np.ones(len(dataset), dtype=bool)
    mask[remove_idx] = False
    dataset = dataset[mask]
    return dataset


if __name__ == "__main__":

    batch_size = 1
    NUM_WORKERS = 0
    train_split = "train_0"
    valid_split = "valid_0"
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--layer_type",
        required=True,
        type=str,
        choices=["FiLMConv", "GATConv", "GCNConv"],
    )
    parser.add_argument(
        "--model_architecture", required=True, type=str, choices=["aae", "vae"]
    )
    parser.add_argument("--use_oclr_scheduler", action="store_true")
    parser.add_argument("--using_cyclical_anneal", action="store_true")
    parser.add_argument("--using_wasserstein_loss", action="store_true")
    parser.add_argument("--use_clamp_log_var", action="store_true")
    parser.add_argument("--using_gp", action="store_true")
    parser.add_argument("--gradient_clip_val", required=True, type=float, default=1.0)
    parser.add_argument("--max_lr", required=True, type=float, default=1e-5)
    parser.add_argument(
        "--gen_step_drop_probability", required=True, type=float, default=0.5
    )
    parser.add_argument("--pretrained_ckpt", type=str)
    parser.add_argument("--pretrained_ckpt_model_type", type=str)
    parser.add_argument(
        "--config_file", type=str, default="config/ldm_con_mic+wae_con.yml"
    )
    parser.add_argument(
        "--raw_trace_dir",
        type=str,
        required=True,
        help="Parent folder containing train_0/valid_0/test_0 trace subfolders "
             "(output of the MoLeR preprocess CLI on the MIC dataset).",
    )
    parser.add_argument(
        "--output_pyg_dir",
        type=str,
        required=True,
        help="Folder where processed PyG .pt shards + processed_file_paths.csv "
             "will be read from / written to.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Use 'cpu' for local smoke tests on machines without a GPU "
             "(e.g. WSL dev machine), 'gpu' for real training on Bionet03.",
    )
    parser.add_argument(
        "--gpu_device",
        type=int,
        default=0,
        help="Which GPU index to use when --accelerator=gpu. "
             "On Bionet03, only 0/1/3 are usable -- do not use GPU 2.",
    )

    '''
    Smoke test on the 20-molecule mock dataset on a CPU-only dev machine
    (no convergence expected, just verifying the forward/backward pass runs
    end to end with dim=1 condition and the repackaged first-stage checkpoint):

    python train_ldm_mic.py --layer_type=FiLMConv --model_architecture=aae \
        --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 \
        --using_wasserstein_loss --using_gp --gen_step_drop_probability=0.9 \
        --config_file=config/ldm_con_mic+wae_con.yml \
        --raw_trace_dir=/mnt/d/projects/antibiotics/data/moler_trace_test \
        --output_pyg_dir=/mnt/d/projects/antibiotics/data/moler_pyg_test \
        --max_epochs=2 --accelerator=cpu

    Real WAE (conditional, MIC) training on Bionet03 (GPU 0/1/3 only, never 2),
    once the smoke test passes and the full ChEMBL trace dataset has been
    preprocessed:

    python train_ldm_mic.py --layer_type=FiLMConv --model_architecture=aae \
        --use_oclr_scheduler --gradient_clip_val=1.0 --max_lr=1e-4 \
        --using_wasserstein_loss --using_gp --gen_step_drop_probability=0.9 \
        --config_file=config/ldm_con_mic+wae_con.yml \
        --raw_trace_dir=<full MIC trace dir> \
        --output_pyg_dir=<full MIC pyg output dir> \
        --accelerator=gpu --gpu_device=0
    '''

    args = parser.parse_args()

    config = OmegaConf.load(args.config_file)
    ldm_params = config['model']['params']
    log_name = args.config_file.split('/')[-1].split('.')[0]

    train_dataset = MolerDataset(
        root=args.output_pyg_dir,
        raw_moler_trace_dataset_parent_folder=args.raw_trace_dir,
        output_pyg_trace_dataset_parent_folder=args.output_pyg_dir,
        split=train_split,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )

    valid_dataset = MolerDataset(
        root=args.output_pyg_dir,
        raw_moler_trace_dataset_parent_folder=args.raw_trace_dir,
        output_pyg_trace_dataset_parent_folder=args.output_pyg_dir,
        split=valid_split,
        gen_step_drop_probability=args.gen_step_drop_probability,
    )

    # train_dataset = filter_dataset(791, train_dataset)
    # valid_dataset = filter_dataset(791, valid_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        follow_batch=[
            "correct_edge_choices",
            "correct_edge_types",
            "valid_edge_choices",
            "valid_attachment_point_choices",
            "correct_attachment_point_choice",
            "correct_node_type_choices",
            "original_graph_x",
            "correct_first_node_type_choices",
        ],
        num_workers=NUM_WORKERS,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=[
            "correct_edge_choices",
            "correct_edge_types",
            "valid_edge_choices",
            "valid_attachment_point_choices",
            "correct_attachment_point_choice",
            "correct_node_type_choices",
            "original_graph_x",
            "correct_first_node_type_choices",
        ],
        num_workers=NUM_WORKERS,
    )

    first_stage_params = get_params(train_dataset)
    ###################################################
    first_stage_params["full_graph_encoder"]["layer_type"] = args.layer_type
    first_stage_params["partial_graph_encoder"]["layer_type"] = args.layer_type
    first_stage_params["use_oclr_scheduler"] = args.use_oclr_scheduler
    first_stage_params["using_cyclical_anneal"] = args.using_cyclical_anneal
    model_architecture = args.model_architecture
    first_stage_params["max_lr"] = args.max_lr
    ###################################################
    first_stage_config = config['model']['first_stage_config']

    if model_architecture == "aae":
        # Kept identical to train_ldm_l1000.py on purpose -- see module
        # docstring at top of this file for why this is safe to leave alone.
        first_stage_params["gene_exp_condition_mlp"]["input_feature_dim"] = 832 + 978 + 1

    ldm_model = LatentDiffusion(
        first_stage_config,
        config['model']['cond_stage_config'],
        train_dataset,
        args.gen_step_drop_probability,
        batch_size,
        first_stage_params,
        first_stage_config['ckpt_path'],
        unet_config=config['model']['unet_config'],
        **ldm_params
    )

    lr = config.model.base_learning_rate
    ldm_model.learning_rate = lr

    # Get current time for folder path.
    now = str(datetime.now()).replace(" ", "_").replace(":", "_")

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"lightning_logs/{now}", name=f"logs_mic_{now}"
    )
    early_stopping = EarlyStopping(monitor=ldm_params.monitor, patience=3)
    if model_architecture == "vae" or model_architecture == "aae":
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val/loss",
            dirpath=f"lightning_logs/mic_{log_name}_{now}",
            mode="min",
            filename='epoch={epoch:02d}-val_loss={val/loss:.2f}',
            auto_insert_metric_name=False,
        )
    else:
        raise NotImplementedError('model_architecture must be either "vae" or "aae"')

    callbacks = [checkpoint_callback, lr_monitor, early_stopping]

    trainer = Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        devices=[args.gpu_device] if args.accelerator == "gpu" else 1,
        callbacks=callbacks,
        logger=tensorboard_logger,
        gradient_clip_val=args.gradient_clip_val,
    )
    trainer.fit(ldm_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
