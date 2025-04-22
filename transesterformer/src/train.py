############################  src/train.py  ################################
"""
Training loop, loss calculation, evaluation, and checkpoint logic.
Compatible with run_training.py (expects run_training_pipeline(data_path,
model_load_path=None)).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
#  Bootstrap so direct execution finds package modules
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent          # .../transesterformer/src
_ROOT_DIR = _THIS_DIR.parent                         # .../transesterformer
for p in (_THIS_DIR, _ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---------------------------------------------------------------------------
#  Third‑party / stdlib imports
# ---------------------------------------------------------------------------
import os
import torch
import torch.optim as optim
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Internal absolute imports
# ---------------------------------------------------------------------------
from src.constants import (
    DEVICE,
    N_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SAVE_FREQ,
    ODE_SOLVER,
    ODE_TOLERANCE,
    MODEL_SAVE_DIR,
    CONDITION_COLS,
)
from src.model import Transesterformer
from src.data_loader import get_dataloader
from src.utils import (
    denormalize_species,
    denormalize_conditions,
    plot_predictions,
    save_checkpoint,
    load_checkpoint,
    USE_NORMALIZATION,
)

# ---------------------------------------------------------------------------
#  Loss utilities
# ---------------------------------------------------------------------------
def calculate_loss(pred, target, mask, model):
    mask = mask.bool()
    mse = ((pred - target) * mask) ** 2
    data_loss = torch.sum(mse) / torch.sum(mask).clamp(min=1)
    total_loss = data_loss + model.neural_reg_loss
    return total_loss, data_loss

# ---------------------------------------------------------------------------
#  Training epoch
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimiser, epoch: int):
    model.train()
    running_total = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [train]", leave=False)

    for batch in pbar:
        optimiser.zero_grad()
        batch_loss = 0.0
        n_ok = 0

        for exp in batch:
            if exp["times"][0] != 0:
                continue  # skip malformed experiment

            preds = model(
                exp["initial_conditions"].unsqueeze(0),
                exp["times"],
                exp["conditions"].unsqueeze(0),
            )
            loss, _ = calculate_loss(
                preds,
                exp["species_norm"].unsqueeze(0),
                exp["mask"].unsqueeze(0),
                model,
            )
            loss.backward()
            batch_loss += loss.item()
            n_ok += 1

        if n_ok:
            optimiser.step()
            running_total += batch_loss
            pbar.set_postfix({"loss": batch_loss / n_ok})

    pbar.close()
    n_samples = len(loader.dataset)
    return running_total / n_samples if n_samples else float("inf")

# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, epoch: int, n_plot: int = 5):
    model.eval()
    tot = 0.0
    plotted = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for batch in pbar:
            batch_loss = 0.0
            n_ok = 0

            for exp in batch:
                if exp["times"][0] != 0:
                    continue

                preds = model(
                    exp["initial_conditions"].unsqueeze(0),
                    exp["times"],
                    exp["conditions"].unsqueeze(0),
                )
                loss, _ = calculate_loss(
                    preds,
                    exp["species_norm"].unsqueeze(0),
                    exp["mask"].unsqueeze(0),
                    model,
                )
                batch_loss += loss.item()
                n_ok += 1

                if plotted < n_plot:
                    t_np = exp["times"].cpu().numpy()
                    true_np = denormalize_species(exp["species_norm"])
                    pred_np = denormalize_species(preds.squeeze(0))
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
                pbar.set_postfix({"val_loss": batch_loss / n_ok})
                tot += batch_loss

        pbar.close()

    n_samples = len(loader.dataset)
    return tot / n_samples if n_samples else float("inf")

# ---------------------------------------------------------------------------
#  Orchestrator (keeps original keyword: model_load_path)
# ---------------------------------------------------------------------------
def run_training_pipeline(data_path: str, model_load_path: str | None = None):
    print(f"Starting training on {DEVICE}")
    print(f"Normalization enabled: {USE_NORMALIZATION}")

    train_loader = get_dataloader(data_path=data_path, shuffle=True)
    val_loader = get_dataloader(data_path=data_path, shuffle=False)

    model = Transesterformer(
        n_conditions=len(CONDITION_COLS),
        ode_solver=ODE_SOLVER,
        ode_options={"rtol": ODE_TOLERANCE, "atol": ODE_TOLERANCE},
    ).to(DEVICE)

    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch, best_val = 0, float("inf")
    if model_load_path and os.path.exists(model_load_path):
        start_epoch, best_val = load_checkpoint(model, optimiser, model_load_path, DEVICE)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, N_EPOCHS):
        _ = train_epoch(model, train_loader, optimiser, epoch)
        val_loss = evaluate(model, val_loader, epoch)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model,
                optimiser,
                epoch,
                val_loss,
                os.path.join(MODEL_SAVE_DIR, "best_model.pth"),
            )

        if (epoch + 1) % SAVE_FREQ == 0:
            save_checkpoint(
                model,
                optimiser,
                epoch,
                val_loss,
                os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth"),
            )

    print(f"Training finished. Best validation loss: {best_val:.6f}")

# ---------------------------------------------------------------------------
#  Direct execution helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    default_data = (_ROOT_DIR / "data" / "processed" / "kinetic_data.csv").as_posix()
    parser.add_argument("--data", default=default_data)
    parser.add_argument("--ckpt", default=None, dest="model_load_path")
    args = parser.parse_args()

    run_training_pipeline(args.data, args.model_load_path)
