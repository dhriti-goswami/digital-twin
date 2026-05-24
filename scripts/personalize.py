#!/usr/bin/env python3
"""
Personalization Script for Diabetes Digital Twin.

Fine-tunes the general population model on a specific patient's time series
data to create a personalized clinical digital twin.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model import GlucoseDataset, load_data, prepare_patient_data
from src.models.glucose_predictor import GlucosePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def personalize_patient_model(
    patient_id: int,
    data_dir: Path,
    base_model_path: Path,
    output_dir: Path,
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    """
    Fine-tunes the population model on a single patient's data.
    """
    logger.info(f"=== Personalizing model for Patient {patient_id} ===")
    
    # 1. Load trained population model checkpoint
    if not base_model_path.exists():
        logger.error(f"Base model checkpoint not found: {base_model_path}")
        return False
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)
        config = checkpoint.get("config")
        feature_names = checkpoint.get("feature_names", None)
    except Exception as e:
        logger.error(f"Failed to load population base model: {e}")
        return False
        
    # 2. Load and filter patient data
    glucose_df, insulin_df, meals_df = load_data(data_dir)
    
    # Extract feature engine from model run or create a new matching one
    from src.data.preprocessing import GlucoseFeatureEngine
    feature_engine = GlucoseFeatureEngine(
        sequence_length=24,
        prediction_horizons=[6, 12, 18, 24],
        cgm_interval_minutes=5,
    )
    
    # Load clinical profiles for static covariate mappings
    pima_path = PROJECT_ROOT / "data" / "raw" / "pima" / "pima_diabetes.csv"
    hosp_path = PROJECT_ROOT / "data" / "raw" / "diabetes_130_hospitals"
    
    from scripts.train_model import load_static_clinical_profiles
    pima_profiles, hosp_profiles = load_static_clinical_profiles(pima_path, hosp_path)
    
    pima_idx = (patient_id * 13) % len(pima_profiles)
    hosp_idx = (patient_id * 101) % len(hosp_profiles)
    pima_p = pima_profiles.iloc[pima_idx]
    hosp_p = hosp_profiles.iloc[hosp_idx]
    
    # Prepare patient's individual data
    result = prepare_patient_data(
        patient_id, glucose_df, insulin_df, meals_df, feature_engine, pima_p, hosp_p
    )
    
    if result is None:
        logger.warning(f"Patient {patient_id} does not have sufficient history for personalization")
        return False
        
    X, y, names, gh, iob, cob = result
    logger.info(f"Loaded {len(X)} individual sequences for Patient {patient_id}")
    
    # Fit the patient's individual scaling
    X_flat = X.reshape(-1, X.shape[2])
    feature_engine.scaler.fit(X_flat)
    feature_engine._is_fitted = True
    X_scaled = feature_engine.scaler.transform(X_flat).reshape(X.shape).astype(np.float32)
    
    # Create DataLoader
    dataset = GlucoseDataset(X_scaled, y, gh, iob, cob)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize model and load base weights
    state_dict = checkpoint["model_state_dict"]
    input_size = X_scaled.shape[2]
    num_horizons = y.shape[1]
    
    model = GlucosePredictor(
        input_size=input_size,
        model_type=config.model_type,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_horizons=num_horizons,
        use_pinn=config.use_pinn,
        pinn_lambda=config.pinn_lambda,
    ).to(device)
    
    model.load_state_dict(state_dict)
    
    # 4. Freeze core feature extraction layers to preserve general population properties
    # Fine-tunes only the high-level personalized projection heads
    frozen_count = 0
    trainable_count = 0
    
    if config.model_type == "transformer":
        # Freeze transformer encoder and input embedding
        for name, param in model.model.input_embedding.named_parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        for name, param in model.model.positional_encoding.named_parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        for name, param in model.model.transformer_encoder.named_parameters():
            param.requires_grad = False
            frozen_count += param.numel()
            
        # Keep projection heads trainable
        for name, param in model.model.fc1.named_parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        for name, param in model.model.fc2.named_parameters():
            param.requires_grad = True
            trainable_count += param.numel()
            
    elif config.model_type == "lstm":
        # Freeze LSTM and projection layers
        for name, param in model.model.input_proj.named_parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        for name, param in model.model.lstm.named_parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        for name, param in model.model.attention.named_parameters():
            param.requires_grad = False
            frozen_count += param.numel()
            
        # Keep output heads trainable
        for name, param in model.model.fc1.named_parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        for name, param in model.model.fc2.named_parameters():
            param.requires_grad = True
            trainable_count += param.numel()
            
    logger.info(f"Frozen parameters: {frozen_count:,} | Trainable (fine-tuned) parameters: {trainable_count:,}")
    
    # 5. Fine-tune model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mse = 0
        epoch_physics = 0
        batches = 0
        
        for batch in loader:
            batch_X, batch_y, batch_gh, batch_iob, batch_cob = batch
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_gh = batch_gh.to(device)
            batch_iob = batch_iob.to(device)
            batch_cob = batch_cob.to(device)
            
            optimizer.zero_grad()
            
            pred = model(batch_X)
            loss, loss_dict = model.compute_loss(pred, batch_y, batch_gh, batch_iob, batch_cob)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss_dict["total_loss"]
            epoch_mse += loss_dict["mse_loss"]
            epoch_physics += loss_dict.get("physics_loss", 0.0)
            batches += 1
            
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"  Epoch {epoch+1:2d}/{epochs:2d} │ "
                f"Loss: {epoch_loss/batches:.4f} │ "
                f"MSE: {epoch_mse/batches:.4f} │ "
                f"Physics Loss: {epoch_physics/batches:.4f}"
            )
            
    # 6. Save patient-specific checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"patient_{patient_id}_model.pt"
    
    # Save feature engine inside the checkpoint so the feature_engine is fully portable
    torch.save({
        "patient_id": patient_id,
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_names": names,
        "feature_engine": feature_engine,
        "personalization_date": pd.Timestamp.now().isoformat(),
        "parent_checkpoint": str(base_model_path),
    }, output_path)
    
    logger.info(f"✓ Successfully personalized digital twin model saved to: {output_path}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Personalize general T1D model per patient")
    parser.add_argument("--patient-id", type=int, required=True, help="ID of patient to personalize for")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Data directory path")
    parser.add_argument("--base-model", type=str, default="./checkpoints/best_model.pt", help="Path to base model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Where to save personalized models")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    args = parser.parse_args()

    personalize_patient_model(
        patient_id=args.patient_id,
        data_dir=Path(args.data_dir),
        base_model_path=Path(args.base_model),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
