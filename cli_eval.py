#!/usr/bin/env python3
"""
Cadenza Post-Competition Experiment - CLI Training Script

Usage:
    python cadenza_train_cli.py --epochs 2 --use-weighted-mean
    python cadenza_train_cli.py --epochs 5 --no-weighted-mean --learning-rate 1e-4
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
import json
import shutil
from typing import Optional

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.model_selection import KFold
import safetensors.torch

from transformers import (
    WhisperModel, WhisperConfig, WhisperPreTrainedModel,
    AutoModel, AutoConfig, AutoFeatureExtractor,
    PreTrainedModel, TrainingArguments, Trainer, set_seed
)
import transformers
import warnings

warnings.simplefilter("ignore")

# Global constants
SR = 16000
MODEL_ID = "openai/whisper-medium.en"
FT_MODEL_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _compute_mask_indices(
    shape: tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for SpecAugment.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    epsilon = np.random.rand(1).item()
    
    def compute_num_masked_span(input_length):
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    input_lengths = (
        attention_mask.detach().sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        num_masked_span = compute_num_masked_span(input_length)

        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        if len(spec_aug_mask_idx) == 0:
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class EncoderForIntel(PreTrainedModel):
    def __init__(self, config, model_id):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_id).encoder

    def forward(self, input_features, output_hidden_states=True, return_dict=True):
        encoder_outputs = self.encoder(
            input_features,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return encoder_outputs


class IntelDataset(data.Dataset):
    def __init__(self, model_id, meta_df, training=False, unproc=True, use_parakeet=False):
        self.feat_extractor = AutoFeatureExtractor.from_pretrained(
            model_id, feature_size=128
        ) if use_parakeet else AutoFeatureExtractor.from_pretrained(model_id)
        self.meta_df = meta_df
        self.hloss = {"No Loss": 0, "Mild": 1, "Moderate": 2}
        self.unproc = unproc
        self.use_parakeet = use_parakeet
        self.training = training

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        file_path = os.path.join(
            self.meta_df.iloc[idx]["path"],
            self.meta_df.iloc[idx]["signal"] + self.meta_df.iloc[idx]["extension"]
        )

        audio = librosa.load(file_path, sr=SR)
        level = torch.tensor(
            self.hloss[self.meta_df.iloc[idx]["hearing_loss"]], dtype=torch.float32
        ) if self.unproc else None

        if self.feat_extractor.feature_extractor_type == "WhisperFeatureExtractor":
            if self.use_parakeet:
                self.feat_extractor.feature_size = 128
            features = self.feat_extractor(
                audio[0], sampling_rate=SR, return_tensors="pt"
            )
            if self.training:
                data = {
                    "input_features": features.input_features.squeeze(),
                    "labels": torch.tensor(self.meta_df.iloc[idx]["correctness"], dtype=torch.float32),
                    "level": level
                }
            else:
                data = {
                    "input_features": features.input_features.squeeze(),
                    "fname": self.meta_df.iloc[idx]["signal"],
                    "level": level
                }
            return data

        features = self.feat_extractor(
            audio[0], sampling_rate=SR, return_tensors="pt",
            padding='max_length', max_length=400000
        )
        if self.training:
            data = {
                "input_values": features.input_values[0].squeeze(),
                "labels": torch.tensor(self.meta_df.iloc[idx]["correctness"], dtype=torch.float32),
                "level": level
            }
        else:
            data = {
                "input_values": features.input_values[0].squeeze(),
                "fname": self.meta_df.iloc[idx]["signal"],
                "level": level
            }
        return data


class TfmModel(PreTrainedModel):
    def __init__(self, config, encoder, finetune_config, use_weighted_mean, unproc, layer, do_norm):
        super().__init__(config)
        self.encoder = encoder
        self.config = finetune_config
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.tfm = WhisperModel(finetune_config).encoder
        self.loss_embedding = nn.Embedding(3, finetune_config.hidden_size)
        self.unproc = unproc
        self.proj_input_feats = nn.Linear(config.hidden_size, finetune_config.hidden_size)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(finetune_config.max_source_positions)

        self.output_size = 2 * finetune_config.hidden_size if self.unproc else finetune_config.hidden_size
        self.proj = nn.Linear(self.output_size, self.output_size)
        self.out = nn.Linear(self.output_size, 1)
        self.use_weighted_mean = use_weighted_mean
        num_layers = self.encoder.config.num_hidden_layers + 1
        if self.use_weighted_mean:
            self.alpha = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.layer = layer
        self.do_norm = do_norm

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

    def forward(self, input_features, level=None, labels=None, output_hidden_states=True, return_dict=True):
        loss_fxn = nn.MSELoss()
        encoder_outputs = self.encoder(
            input_features, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        if self.use_weighted_mean:
            hidden_states = encoder_outputs.hidden_states
            if self.do_norm:
                hidden_states = [self.encoder.encoder.layer_norm(hidden_state) for hidden_state in hidden_states]
            hidden_states = torch.stack(hidden_states, dim=1)
            hidden_states = (hidden_states * self.alpha.view(-1, 1, 1)).sum(dim=1)
            input_features = hidden_states.permute(0, 2, 1)
        else:
            if self.layer is None:
                self.layer = -1
            input_features = (
                encoder_outputs.last_hidden_state if self.layer == -1
                else self.encoder.encoder.layer_norm(encoder_outputs.hidden_states[self.layer])
            )
            input_features = input_features.permute(0, 2, 1)

        input_features = self._mask_input_features(input_features)
        tfm_outputs = self.tfm(input_features, return_dict=return_dict)

        out = tfm_outputs.last_hidden_state
        out = nn.ReLU()(self.proj(out))
        out = out.mean(dim=1)

        if self.unproc:
            loss_emb = self.loss_embedding(level.long())
            out = torch.cat((tfm_outputs.last_hidden_state.mean(dim=1), loss_emb), dim=-1)
        out = self.out(out)
        out = nn.Sigmoid()(out)

        if labels is not None:
            loss = loss_fxn(out, labels.view(-1, 1))
            return {"logits": out, "loss": loss}
        else:
            return {"logits": out}


def load_metadata(data_dir):
    """Load and prepare metadata dataframes."""
    train_data_dir = os.path.join(data_dir, "train/signals/")
    
    with open(os.path.join(data_dir, "metadata/train_metadata_with_phoneme.json"), "r") as f:
        meta_train = json.load(f)
    
    train_meta_df = pd.DataFrame(meta_train)
    train_path_df = pd.DataFrame([{"path": train_data_dir} for _ in range(len(train_meta_df))])
    train_extension_df = pd.DataFrame([{"extension": ".flac"} for _ in range(len(train_meta_df))])
    
    train_meta_df = pd.concat([train_path_df, train_meta_df, train_extension_df], axis=1)
    
    return train_meta_df


def train_kfold(args):
    """Main training function with k-fold cross-validation."""
    print(f"Using device: {DEVICE}")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Use weighted mean: {args.use_weighted_mean}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Output directory: {args.output_dir}")
    
    # Load metadata
    train_meta_df = load_metadata(args.data_dir)
    print(f"Loaded {len(train_meta_df)} training samples")
    
    # Load encoder model
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = EncoderForIntel(config, MODEL_ID)
    
    # Configure fine-tuning model
    config_ft = AutoConfig.from_pretrained(FT_MODEL_ID)
    config_ft.num_mel_bins = 1024
    config_ft.max_source_positions = 750
    config_ft.apply_spec_augment = True
    config_ft.mask_time_prob = 0.00
    config_ft.mask_feature_prob = 0.001
    
    # K-Fold cross-validation
    kfold = KFold(n_splits=args.n_folds)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_meta_df)):
        print(f"\n{'='*60}")
        print(f"Training on Fold {fold_idx}")
        print(f"{'='*60}\n")
        
        # Create datasets
        trn_sample = IntelDataset(MODEL_ID, train_meta_df.iloc[train_idx], training=True, unproc=args.unproc)
        val_sample = IntelDataset(MODEL_ID, train_meta_df.iloc[val_idx], training=True, unproc=args.unproc)
        
        # Create model
        tfm_model = TfmModel(config, model, config_ft, args.use_weighted_mean, args.unproc, None, args.do_norm)
        
        # Training arguments
        output_dir = os.path.join(args.output_dir, f"model_fold_{fold_idx}")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="best",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            metric_for_best_model="loss",
        )
        
        # Create trainer
        trainer = Trainer(
            model=tfm_model,
            args=training_args,
            train_dataset=trn_sample,
            eval_dataset=val_sample
        )
        
        # Train
        trainer.train()
        
        # Clean up old checkpoints
        saved_indices = sorted([
            int(name.split("-")[-1])
            for name in os.listdir(output_dir)
            if "checkpoint" in name
        ])
        for idx in saved_indices[:-1]:
            shutil.rmtree(f"{output_dir}/checkpoint-{idx}")
        
        print(f"\nCompleted training for fold {fold_idx}")
    
    print(f"\n{'='*60}")
    print(f"Training completed for all {args.n_folds} folds!")
    print(f"{'='*60}")


def predict_on_dataset(model, dataset, device):
    """Run predictions on a dataset."""
    model.eval()
    model.to(device)
    results = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            out = model(sample["input_features"].unsqueeze(0).to(device))
            results.append({
                "fname": sample["fname"],
                "pred": out["logits"].item()
            })
    
    return pd.DataFrame(results)


def evaluate_models(args):
    """Evaluate all trained models on validation and evaluation sets."""
    print(f"\n{'='*60}")
    print("Starting Model Evaluation")
    print(f"{'='*60}\n")
    
    # Load validation metadata
    valid_dir = os.path.join(args.data_dir, "valid/signals")
    eval_dir = os.path.join(args.data_dir, "eval/signals")
    
    with open(os.path.join(args.data_dir, "metadata/valid_metadata_with_phoneme.json"), "r") as f:
        meta_valid = json.load(f)
    
    with open(os.path.join(args.data_dir, "metadata/eval_metadata_with_phoneme.json"), "r") as f:
        meta_eval = json.load(f)
    
    # Prepare validation dataframe
    val_meta_df = pd.DataFrame(meta_valid)
    valid_path_df = pd.DataFrame([{"path": valid_dir} for _ in range(len(val_meta_df))])
    valid_extension_df = pd.DataFrame([{"extension": ".flac"} for _ in range(len(val_meta_df))])
    val_meta_df = pd.concat([valid_path_df, val_meta_df, valid_extension_df], axis=1)
    
    # Prepare evaluation dataframe
    eval_meta_df = pd.DataFrame(meta_eval)
    eval_path_df = pd.DataFrame([{"path": eval_dir} for _ in range(len(eval_meta_df))])
    eval_extension_df = pd.DataFrame([{"extension": ".flac"} for _ in range(len(eval_meta_df))])
    eval_meta_df = pd.concat([eval_path_df, eval_meta_df, eval_extension_df], axis=1)
    
    # Create datasets
    validation_ds = IntelDataset(MODEL_ID, val_meta_df, training=False, unproc=args.unproc)
    eval_ds = IntelDataset(MODEL_ID, eval_meta_df, training=False, unproc=args.unproc)
    
    print(f"Validation samples: {len(validation_ds)}")
    print(f"Evaluation samples: {len(eval_ds)}")
    
    # Find best checkpoints
    best_ckpts = []
    for i in range(args.n_folds):
        folder_path = os.path.join(args.output_dir, f"model_fold_{i}")
        if not os.path.exists(folder_path):
            print(f"Warning: Fold {i} not found at {folder_path}")
            continue
        folder = os.listdir(folder_path)
        for f in folder:
            if "checkpoint" in f:
                best_ckpts.append((i, f))
                break
    
    if len(best_ckpts) == 0:
        print("Error: No trained models found!")
        return
    
    print(f"\nFound {len(best_ckpts)} trained models")
    
    # Load encoder config
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = EncoderForIntel(config, MODEL_ID)
    
    # Configure fine-tuning model
    config_ft = AutoConfig.from_pretrained(FT_MODEL_ID)
    config_ft.num_mel_bins = 1024
    config_ft.max_source_positions = 750
    config_ft.apply_spec_augment = True
    config_ft.mask_time_prob = 0.00
    config_ft.mask_feature_prob = 0.001
    
    # Collect predictions from all models
    val_results_dict = {}
    eval_results_dict = {}
    
    for fold_idx, ckpt_name in best_ckpts:
        model_name = f"model_{fold_idx}"
        ckpt_path = os.path.join(args.output_dir, f"model_fold_{fold_idx}", ckpt_name)
        
        print(f"\nLoading {model_name} from {ckpt_path}")
        
        tfm_model = TfmModel(config, model, config_ft, args.use_weighted_mean, args.unproc, None, args.do_norm)
        tfm_model.load_state_dict(safetensors.torch.load_file(os.path.join(ckpt_path, "model.safetensors")))
        
        print(f"  Predicting on validation set...")
        val_preds = predict_on_dataset(tfm_model, validation_ds, DEVICE)
        val_results_dict[model_name] = val_preds
        
        print(f"  Predicting on evaluation set...")
        eval_preds = predict_on_dataset(tfm_model, eval_ds, DEVICE)
        eval_results_dict[model_name] = eval_preds
    
    # Combine predictions from all models
    print("\nCombining predictions from all models...")
    
    # Validation predictions
    val_dfs = [df.rename(columns={"pred": f"pred_{i}"}) for i, df in enumerate(val_results_dict.values())]
    val_res_df = val_dfs[0][["fname"]].copy()
    for i, df in enumerate(val_dfs):
        val_res_df[f"pred_{i}"] = df[f"pred_{i}"]
    
    pred_cols = [col for col in val_res_df.columns if col.startswith("pred_")]
    val_res_df["pred"] = val_res_df[pred_cols].mean(axis=1)
    val_res_df = val_res_df[["fname", "pred"]].rename(columns={"fname": "signal"})
    
    # Evaluation predictions
    eval_dfs = [df.rename(columns={"pred": f"pred_{i}"}) for i, df in enumerate(eval_results_dict.values())]
    eval_res_df = eval_dfs[0][["fname"]].copy()
    for i, df in enumerate(eval_dfs):
        eval_res_df[f"pred_{i}"] = df[f"pred_{i}"]
    
    pred_cols = [col for col in eval_res_df.columns if col.startswith("pred_")]
    eval_res_df["pred"] = eval_res_df[pred_cols].mean(axis=1)
    eval_res_df = eval_res_df[["fname", "pred"]].rename(columns={"fname": "signal"})
    
    # Calculate RMSE
    val_rmse = np.sqrt(np.mean((val_res_df.pred - val_meta_df.correctness) ** 2))
    eval_rmse = np.sqrt(np.mean((eval_res_df.pred - eval_meta_df.correctness) ** 2))
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Validation RMSE: {val_rmse:.6f}")
    print(f"Evaluation RMSE: {eval_rmse:.6f}")
    print(f"{'='*60}\n")
    
    # Save predictions
    val_output_path = os.path.join(args.output_dir, "validation_predictions.csv")
    eval_output_path = os.path.join(args.output_dir, "evaluation_predictions.csv")
    
    val_res_df.to_csv(val_output_path, index=False)
    eval_res_df.to_csv(eval_output_path, index=False)
    
    print(f"Predictions saved to:")
    print(f"  - {val_output_path}")
    print(f"  - {eval_output_path}")
    
    # Save results summary
    results_summary = {
        "validation_rmse": float(val_rmse),
        "evaluation_rmse": float(eval_rmse),
        "n_models": len(best_ckpts),
        "use_weighted_mean": args.use_weighted_mean,
        "epochs": args.epochs,
    }
    
    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"  - {summary_path}\n")
    
    return val_rmse, eval_rmse


def main():
    parser = argparse.ArgumentParser(
        description="Cadenza Post-Competition Experiment Training Script"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "evaluate", "both"],
        help="Mode: 'train', 'evaluate', or 'both' (default: train)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--use-weighted-mean", dest="use_weighted_mean", action="store_true",
        help="Use weighted mean of hidden states"
    )
    parser.add_argument(
        "--no-weighted-mean", dest="use_weighted_mean", action="store_false",
        help="Don't use weighted mean of hidden states"
    )
    parser.set_defaults(use_weighted_mean=False)
    parser.add_argument(
            "--no-norm", dest="do_norm", action="store_false",
            help="Do not norm the hidden states"
    )
    parser.set_defaults(do_norm=True)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=500,
        help="Number of warmup steps (default: 500)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.001,
        help="Weight decay (default: 0.001)"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    
    # Data parameters
    parser.add_argument(
        "--data-dir", type=str, default="cadenza_data",
        help="Path to data directory (default: cadenza_data)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Path to output directory (default: ./results)"
    )
    parser.add_argument(
        "--unproc", action="store_true", default=False,
        help="Use unprocessed data with hearing loss embedding"
    )
    
    # Other parameters
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run based on mode
    if args.mode in ["train", "both"]:
        train_kfold(args)
    
    if args.mode in ["evaluate", "both"]:
        evaluate_models(args)


if __name__ == "__main__":
    main()
