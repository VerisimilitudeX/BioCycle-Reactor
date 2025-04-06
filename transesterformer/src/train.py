# src/train.py
"""
Contains the main training loop, loss calculation, validation, and plotting logic.
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import os
from tqdm import tqdm # Progress bar

from .constants import (
    DEVICE, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PRINT_FREQ, SAVE_FREQ,
    ODE_SOLVER, ODE_TOLERANCE, LAMBDA_MASS_BALANCE,
    MODEL_SAVE_DIR, FIGURE_SAVE_DIR, N_SPECIES
)
from .model import Transesterformer
from .data_loader import get_dataloader
from .utils import (
    denormalize_species, denormalize_conditions, plot_predictions,
    save_checkpoint, load_checkpoint, USE_NORMALIZATION
)

def calculate_loss(predictions, targets, mask, model):
    """
    Calculates the loss, considering masking for missing values and physics constraints.

    Args:
        predictions (torch.Tensor): Model output (batch_size, n_times, n_species)
        targets (torch.Tensor): Ground truth species data (batch_size, n_times, n_species)
                                (NaNs filled with 0, use mask)
        mask (torch.Tensor): Boolean mask indicating valid data points
                             (batch_size, n_times, n_species)
        model (nn.Module): The Transesterformer model instance (to access reg loss).

    Returns:
        torch.Tensor: The total calculated loss.
        torch.Tensor: The data fidelity loss (MSE on valid points).
    """
    # Ensure mask is boolean
    mask = mask.bool()

    # 1. Data Fidelity Loss (Masked MSE)
    # Calculate squared error only for valid points indicated by the mask
    error = predictions - targets
    masked_error = error * mask # Zero out errors for invalid points
    
    # Calculate MSE only over valid points
    # Sum of squared errors over valid points / number of valid points
    loss_data = torch.sum(masked_error**2) / torch.sum(mask).clamp(min=1) # Avoid division by zero if mask is all False

    # 2. Physics Regularization Loss (from ODE function)
    loss_reg_neural = model.neural_reg_loss # Get the stored reg loss

    # 3. Optional: Mass Balance Penalty (if not structurally enforced)
    # This requires calculating total moles of fatty acid chains at each time step
    # Example: Check if sum(TG*3 + DG*2 + MG*1 + FAME*1) is constant
    loss_mass_balance = torch.tensor(0.0, device=DEVICE)
    if LAMBDA_MASS_BALANCE > 0:
         # This calculation needs denormalized values and initial moles
         # It's complex to implement correctly here, especially with normalization.
         # For now, assume mass balance is structurally enforced or ignore this term.
         pass # Placeholder for mass balance calculation


    # 4. Total Loss
    total_loss = loss_data + loss_reg_neural + LAMBDA_MASS_BALANCE * loss_mass_balance

    return total_loss, loss_data


def train_epoch(model, dataloader, optimizer, epoch):
    """Runs one epoch of training."""
    model.train()
    total_loss_epoch = 0.0
    total_data_loss_epoch = 0.0
    start_time = time.time()

    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Training]", leave=False)

    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        batch_loss_total = torch.tensor(0.0, device=DEVICE)
        batch_loss_data = torch.tensor(0.0, device=DEVICE)
        n_samples_in_batch = len(batch)

        # Process each experiment in the batch individually
        # (because odeint typically handles batches, but our data structure is list of dicts)
        # Alternatively, pad sequences and batch properly if performance is critical.
        for exp_data in batch:
            initial_cond = exp_data['initial_conditions'].unsqueeze(0) # Add batch dim of 1
            times = exp_data['times']
            conditions = exp_data['conditions'].unsqueeze(0) # Add batch dim of 1
            targets = exp_data['species_norm'].unsqueeze(0) # Add batch dim of 1
            mask = exp_data['mask'].unsqueeze(0) # Add batch dim of 1

            # Ensure times are sorted and start from t=0 if needed by solver
            if times[0] != 0:
                 # This indicates an issue in data prep or assumptions
                 print(f"Warning: Experiment {exp_data['exp_id']} times do not start at 0 ({times[0]}). Adjusting.")
                 # Option 1: Shift times (if relative time is okay)
                 # times = times - times[0]
                 # Option 2: Prepend t=0 (requires knowing C(t=0) accurately)
                 # initial_cond = ... # Need C(t=0)
                 # times = torch.cat([torch.tensor([0.0], device=DEVICE), times])
                 # targets = torch.cat([initial_cond.unsqueeze(1), targets], dim=1) # Add target at t=0
                 # mask = torch.cat([torch.ones_like(initial_cond.unsqueeze(1), dtype=torch.bool), mask], dim=1) # Add mask
                 # For simplicity, skip experiment if times don't start near 0
                 print(f"Skipping experiment {exp_data['exp_id']} due to time issue.")
                 continue


            try:
                # Forward pass
                predictions = model(initial_cond, times, conditions) # Shape (1, n_times, n_species)

                # Calculate loss for this single experiment
                loss_exp, loss_data_exp = calculate_loss(predictions, targets, mask, model)

                # Accumulate loss (average over samples in batch later)
                batch_loss_total += loss_exp
                batch_loss_data += loss_data_exp

            except Exception as e:
                print(f"\nError during forward/loss calculation for exp {exp_data['exp_id']}: {e}")
                print("Skipping this experiment for this batch.")
                # import traceback
                # traceback.print_exc()
                n_samples_in_batch -= 1 # Decrement effective batch size


        if n_samples_in_batch > 0:
            # Average loss over the number of successfully processed samples
            avg_batch_loss = batch_loss_total / n_samples_in_batch
            avg_data_loss = batch_loss_data / n_samples_in_batch

            # Backward pass and optimization step
            avg_batch_loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_epoch += avg_batch_loss.item() * n_samples_in_batch # Accumulate total loss before averaging
            total_data_loss_epoch += avg_data_loss.item() * n_samples_in_batch

            # Update progress bar
            pbar.set_postfix({
                'Batch Loss': f"{avg_batch_loss.item():.4f}",
                'Data Loss': f"{avg_data_loss.item():.4f}"
            })
        else:
            print("Warning: No samples successfully processed in this batch.")


    avg_loss_epoch = total_loss_epoch / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    avg_data_loss_epoch = total_data_loss_epoch / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    epoch_time = time.time() - start_time

    pbar.close() # Close the tqdm progress bar for the epoch
    print(f"Epoch {epoch+1}/{N_EPOCHS} | Avg Train Loss: {avg_loss_epoch:.6f} | Avg Data Loss: {avg_data_loss_epoch:.6f} | Time: {epoch_time:.2f}s")

    return avg_loss_epoch


def evaluate(model, dataloader, epoch, plot_n_examples=5):
    """Evaluates the model on the validation set and generates plots."""
    model.eval()
    total_val_loss = 0.0
    total_val_data_loss = 0.0
    n_plotted = 0

    print(f"\n--- Evaluating Epoch {epoch+1} ---")
    val_start_time = time.time()

    with torch.no_grad():
      pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Validation]", leave=False)
      for i, batch in enumerate(pbar):
            batch_loss_total = torch.tensor(0.0, device=DEVICE)
            batch_loss_data = torch.tensor(0.0, device=DEVICE)
            n_samples_in_batch = len(batch)

            for exp_data in batch:
                initial_cond = exp_data['initial_conditions'].unsqueeze(0)
                times = exp_data['times']
                conditions = exp_data['conditions'].unsqueeze(0)
                targets = exp_data['species_norm'].unsqueeze(0)
                mask = exp_data['mask'].unsqueeze(0)
                exp_id = exp_data['exp_id']

                # Skip if time issue persists (should be caught in training too)
                if times[0] != 0: continue

                try:
                    predictions = model(initial_cond, times, conditions)
                    loss_exp, loss_data_exp = calculate_loss(predictions, targets, mask, model)

                    batch_loss_total += loss_exp
                    batch_loss_data += loss_data_exp

                    # --- Plotting ---
                    if n_plotted < plot_n_examples:
                        # Denormalize for plotting
                        pred_np = denormalize_species(predictions.squeeze(0)) # Remove batch dim
                        true_np = denormalize_species(targets.squeeze(0)) # Remove batch dim
                        times_np = times.cpu().numpy()

                        # Get denormalized conditions for plot title
                        conditions_denorm_tensor = denormalize_conditions(conditions)
                        conditions_denorm_dict = {
                            name: val.item() for name, val in zip(dataloader.dataset.experiments[0]['conditions'].cpu().numpy(), conditions_denorm_tensor.squeeze().cpu().numpy())
                        }
                         # Use actual condition names
                        conditions_denorm_dict_named = {
                            name: conditions_denorm_tensor.squeeze().cpu().numpy()[i]
                            for i, name in enumerate(dataloader.dataset.experiments[0]['conditions'].keys()) # Assuming first exp has keys representative
                        } if hasattr(dataloader.dataset.experiments[0]['conditions'], 'keys') else \
                        { f"Cond_{i}": val.item() for i, val in enumerate(conditions_denorm_tensor.squeeze().cpu().numpy())} # Fallback naming


                        plot_predictions(exp_id, times_np, true_np, pred_np, conditions_denorm_dict_named)
                        n_plotted += 1

                except Exception as e:
                    print(f"\nError during validation/plotting for exp {exp_id}: {e}")
                    n_samples_in_batch -= 1


            if n_samples_in_batch > 0:
                avg_batch_loss = batch_loss_total / n_samples_in_batch
                avg_data_loss = batch_loss_data / n_samples_in_batch
                total_val_loss += avg_batch_loss.item() * n_samples_in_batch
                total_val_data_loss += avg_data_loss.item() * n_samples_in_batch
                pbar.set_postfix({
                    'Val Loss': f"{avg_batch_loss.item():.4f}",
                    'Data Loss': f"{avg_data_loss.item():.4f}"
                 })
            else:
                 print("Warning: No samples successfully evaluated in this validation batch.")


    avg_val_loss = total_val_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    avg_val_data_loss = total_val_data_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    val_time = time.time() - val_start_time
    pbar.close()
    print(f"Epoch {epoch+1}/{N_EPOCHS} | Avg Valid Loss: {avg_val_loss:.6f} | Avg Valid Data Loss: {avg_val_data_loss:.6f} | Eval Time: {val_time:.2f}s")
    print(f"--- Evaluation Finished ---")

    return avg_val_loss


def run_training_pipeline(data_path, model_load_path=None):
    """Main function to orchestrate the training and evaluation pipeline."""

    print(f"Starting training pipeline using device: {DEVICE}")
    print(f"Normalization enabled: {USE_NORMALIZATION}")

    # --- Data Loading ---
    # Split data into train/validation (e.g., 80/20 split by experiment ID)
    # This is simplified: loading all data for both train/val
    # TODO: Implement proper train/val split based on experiment IDs
    print("Loading training data...")
    train_loader = get_dataloader(data_path=data_path, shuffle=True)
    print("Loading validation data...")
    val_loader = get_dataloader(data_path=data_path, shuffle=False) # No shuffle for validation

    if not train_loader.dataset.experiments or not val_loader.dataset.experiments:
         print("Failed to load data. Exiting.")
         return

    # --- Model Initialization ---
    ode_options = {'rtol': ODE_TOLERANCE, 'atol': ODE_TOLERANCE} # Set solver tolerances
    model = Transesterformer(
        n_conditions=len(train_loader.dataset.experiments[0]['conditions']), # Get n_conditions from data
        ode_solver=ODE_SOLVER,
        ode_options=ode_options
    ).to(DEVICE)

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Load Checkpoint (if specified) ---
    start_epoch = 0
    best_val_loss = float('inf')
    if model_load_path and os.path.exists(model_load_path):
        print(f"Loading model from checkpoint: {model_load_path}")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, model_load_path, DEVICE)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")


    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, N_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        val_loss = evaluate(model, val_loader, epoch, plot_n_examples=5) # Plot 5 examples

        # --- Save Model Checkpoint ---
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)
            print(f"*** New best model saved with validation loss: {best_val_loss:.6f} ***")

        if (epoch + 1) % SAVE_FREQ == 0:
            save_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)

    print("\n--- Training Finished ---")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

