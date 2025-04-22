# src/train.py
"""
Contains the main training loop, loss calculation, validation, and plotting logic.
Only the imports near the top and the evaluate() function changed.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import os
from tqdm import tqdm

from .constants import (
    DEVICE,
    N_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PRINT_FREQ,
    SAVE_FREQ,
    ODE_SOLVER,
    ODE_TOLERANCE,
    LAMBDA_MASS_BALANCE,
    MODEL_SAVE_DIR,
    FIGURE_SAVE_DIR,
    N_SPECIES,
    CONDITION_COLS,   # new import for tidy plot titles
)
from .model import Transesterformer
from .data_loader import get_dataloader
from .utils import (
    denormalize_species,
    denormalize_conditions,
    plot_predictions,
    save_checkpoint,
    load_checkpoint,
    USE_NORMALIZATION,
)

# ---------------------------------------------------------------------- #
#  calculate_loss and train_epoch remain unchanged                       #
# ---------------------------------------------------------------------- #

def calculate_loss(predictions, targets, mask, model):
    mask = mask.bool()
    error = predictions - targets
    masked_error = error * mask
    loss_data = torch.sum(masked_error ** 2) / torch.sum(mask).clamp(min=1)
    loss_reg_neural = model.neural_reg_loss
    total_loss = loss_data + loss_reg_neural
    return total_loss, loss_data


# ---------------------------------------------------------------------- #
#  evaluate (rewritten for cleaner condition dict and progress messages) #
# ---------------------------------------------------------------------- #
def evaluate(model, dataloader, epoch, plot_n_examples: int = 5):
    model.eval()
    total_val_loss = 0.0
    total_val_data = 0.0
    plotted = 0

    print(f"\nValidation — epoch {epoch + 1}")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in pbar:
            batch_loss = torch.tensor(0.0, device=DEVICE)
            batch_data = torch.tensor(0.0, device=DEVICE)
            n_ok = 0
            for exp in batch:
                if exp["times"][0] != 0:
                    continue  # mis‑formatted experiment

                preds = model(
                    exp["initial_conditions"].unsqueeze(0),
                    exp["times"],
                    exp["conditions"].unsqueeze(0),
                )
                loss, ldata = calculate_loss(
                    preds, exp["species_norm"].unsqueeze(0), exp["mask"].unsqueeze(0), model
                )
                batch_loss += loss
                batch_data += ldata
                n_ok += 1

                # Plot a few examples
                if plotted < plot_n_examples:
                    pred_np = denormalize_species(preds.squeeze(0))
                    true_np = denormalize_species(exp["species_norm"])
                    t_np = exp["times"].cpu().numpy()
                    cond_vals = (
                        denormalize_conditions(exp["conditions"].unsqueeze(0))
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                    cond_dict = {k: float(v) for k, v in zip(CONDITION_COLS, cond_vals)}
                    plot_predictions(exp["exp_id"], t_np, true_np, pred_np, cond_dict)
                    plotted += 1

            if n_ok:
                pbar.set_postfix({"val_loss": (batch_loss / n_ok).item()})
                total_val_loss += batch_loss.item()
                total_val_data += batch_data.item()

    n_samples = len(dataloader.dataset)
    avg_val = total_val_loss / n_samples if n_samples else float("inf")
    print(f"Average validation loss: {avg_val:.6f}")
    return avg_val


# ---------------------------------------------------------------------- #
#  run_training_pipeline remains identical except it imports CONDITION_COLS #
# ---------------------------------------------------------------------- #
def run_training_pipeline(data_path, model_load_path=None):
    print(f"Starting training on device {DEVICE}")
    print(f"Normalization enabled: {USE_NORMALIZATION}")

    train_loader = get_dataloader(data_path=data_path, shuffle=True)
    val_loader = get_dataloader(data_path=data_path, shuffle=False)

    if not train_loader.dataset.experiments:
        print("Failed to load data. Exiting.")
        return

    ode_opts = {"rtol": ODE_TOLERANCE, "atol": ODE_TOLERANCE}
    model = Transesterformer(
        n_conditions=len(CONDITION_COLS), ode_solver=ODE_SOLVER, ode_options=ode_opts
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch, best_val = 0, float("inf")
    if model_load_path and os.path.exists(model_load_path):
        start_epoch, best_val = load_checkpoint(model, optimizer, model_load_path, DEVICE)
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, N_EPOCHS):
        _ = train_epoch(model, train_loader, optimizer, epoch)
        val_loss = evaluate(model, val_loader, epoch)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, os.path.join(MODEL_SAVE_DIR, "best_model.pth")
            )

        if (epoch + 1) % SAVE_FREQ == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    print(f"Training complete. Best validation loss: {best_val:.6f}")
