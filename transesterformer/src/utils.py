# src/utils.py
"""
Utility functions for data processing, normalization, plotting, and saving.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from .constants import (
    SPECIES, SPECIES_MAP, N_SPECIES, MW_OIL, MW_MEOH, MW_FAME, MW_GLY,
    CONDITION_COLS, SPECIES_COLS, TIME_COL, EXP_ID_COL,
    CONDITION_MEANS, CONDITION_STDS, SPECIES_MEANS, SPECIES_STDS, USE_NORMALIZATION,
    FIGURE_SAVE_DIR, MODEL_SAVE_DIR
)

def normalize_conditions(conditions_df):
    """Normalizes condition variables using pre-defined means and stds."""
    if not USE_NORMALIZATION:
        return torch.tensor(conditions_df[CONDITION_COLS].values, dtype=torch.float32)

    conditions_tensor = torch.tensor(conditions_df[CONDITION_COLS].values, dtype=torch.float32)
    # Ensure means and stds are tensors on the correct device
    means = CONDITION_MEANS.to(conditions_tensor.device)
    stds = CONDITION_STDS.to(conditions_tensor.device)
    # Add small epsilon to stds to prevent division by zero
    normalized_conditions = (conditions_tensor - means) / (stds + 1e-8)
    return normalized_conditions

def denormalize_conditions(normalized_conditions_tensor):
    """Denormalizes condition variables."""
    if not USE_NORMALIZATION:
        return normalized_conditions_tensor

    # Ensure means and stds are tensors on the correct device
    means = CONDITION_MEANS.to(normalized_conditions_tensor.device)
    stds = CONDITION_STDS.to(normalized_conditions_tensor.device)
    conditions = normalized_conditions_tensor * (stds + 1e-8) + means
    return conditions

def normalize_species(species_df):
    """Normalizes species concentrations using pre-defined means and stds."""
    if not USE_NORMALIZATION:
        return torch.tensor(species_df[SPECIES_COLS].values, dtype=torch.float32)

    species_tensor = torch.tensor(species_df[SPECIES_COLS].values, dtype=torch.float32)
    means = SPECIES_MEANS.to(species_tensor.device)
    stds = SPECIES_STDS.to(species_tensor.device)
    normalized_species = (species_tensor - means) / (stds + 1e-8)
    return normalized_species

def denormalize_species(normalized_species_tensor):
    """Denormalizes species concentrations."""
    if not USE_NORMALIZATION:
        return normalized_species_tensor.cpu().numpy() # Return numpy array for plotting

    # Ensure tensor is on CPU for numpy conversion if needed
    normalized_species_tensor = normalized_species_tensor.cpu()
    means = SPECIES_MEANS.cpu()
    stds = SPECIES_STDS.cpu()
    species = normalized_species_tensor * (stds + 1e-8) + means
    return species.numpy()

def calculate_initial_moles(oil_mass_g, methanol_oil_ratio, density_oil=0.92, density_methanol=0.79):
    """
    Calculates initial moles of TG and Methanol based on oil mass and ratio.
    Assumes oil mass is given, calculates volume, then moles.
    Args:
        oil_mass_g (float): Initial mass of oil in grams.
        methanol_oil_ratio (float): Molar ratio of methanol to oil (TG).
        density_oil (float): Density of oil (g/mL).
        density_methanol (float): Density of methanol (g/mL).

    Returns:
        tuple: (initial_moles_tg, initial_moles_methanol, total_volume_L)
               Returns NaNs if inputs are invalid.
    """
    if oil_mass_g <= 0 or methanol_oil_ratio <= 0:
        return np.nan, np.nan, np.nan

    moles_tg = oil_mass_g / MW_OIL
    moles_methanol = moles_tg * methanol_oil_ratio

    volume_oil_mL = oil_mass_g / density_oil
    mass_methanol_g = moles_methanol * MW_MEOH
    volume_methanol_mL = mass_methanol_g / density_methanol

    total_volume_L = (volume_oil_mL + volume_methanol_mL) / 1000.0

    if total_volume_L <= 0:
         return np.nan, np.nan, np.nan

    return moles_tg, moles_methanol, total_volume_L

def convert_mass_fraction_to_molar(df, initial_oil_mass_g, density_oil=0.92, density_methanol=0.79):
    """
    Converts mass fractions (%) in a DataFrame to molar concentrations (mol/L).
    Requires initial oil mass and methanol:oil ratio to estimate total volume.
    Assumes the 'methanol_oil_ratio' column exists in the df for the first time point.
    Adds molar concentration columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with mass fraction columns (TG, DG, MG, FAME, Gly)
                           and condition columns including 'methanol_oil_ratio'.
        initial_oil_mass_g (float): The initial mass of oil used for this experiment (e.g., 100g).
                                    This might need to be added based on experiment_id.
                                    *** This is a simplification - real volume changes! ***
        density_oil (float): Density of oil (g/mL).
        density_methanol (float): Density of methanol (g/mL).

    Returns:
        pd.DataFrame: DataFrame with added molar concentration columns.
                      Returns original df if conversion fails.
    """
    # Estimate initial moles and volume using the first time point's ratio
    if df.empty or 'methanol_oil_ratio' not in df.columns:
        print("Warning: Missing 'methanol_oil_ratio' or empty dataframe, cannot convert units.")
        return df

    initial_ratio = df['methanol_oil_ratio'].iloc[0]
    moles_tg_init, moles_meoh_init, total_vol_L = calculate_initial_moles(
        initial_oil_mass_g, initial_ratio, density_oil, density_methanol
    )

    if pd.isna(total_vol_L) or total_vol_L <= 0:
        print(f"Warning: Could not calculate initial volume for experiment. Skipping unit conversion.")
        # Add NaN columns so downstream code doesn't break
        for species in SPECIES:
             if species not in df.columns: # Avoid overwriting if already molar
                 df[species] = np.nan
        return df

    # Assume total mass is roughly conserved (approximation!)
    # Use initial oil mass as reference total mass for % calculation
    total_mass_ref = initial_oil_mass_g # Simplification

    # Molecular weights dictionary
    mw = {'TG': MW_OIL, 'DG': MW_OIL - MW_FAME + MW_MEOH, # Approx DG/MG MWs
          'MG': MW_OIL - 2*MW_FAME + 2*MW_MEOH,
          'FAME': MW_FAME, 'Gly': MW_GLY, 'MeOH': MW_MEOH}

    for species in SPECIES:
        if species in df.columns and species != 'MeOH': # Convert species if column exists
             # Check if data is likely percentage (0-100 range)
             is_percent = df[species].max() <= 100.1 and df[species].min() >= -0.1

             if is_percent:
                 mass_species_g = (df[species] / 100.0) * total_mass_ref
                 moles_species = mass_species_g / mw[species]
                 df[species] = moles_species / total_vol_L # Overwrite with mol/L
             # Else: Assume it's already in mol/L or other unit, leave as is
             # More robust checking could be added here

    # Estimate Methanol concentration (difficult, often not measured directly)
    # We can estimate consumption based on FAME produced
    if 'FAME' in df.columns and 'MeOH' not in df.columns:
        initial_meoh_conc = moles_meoh_init / total_vol_L
        # FAME production consumes MeOH stoichiometrically (3 FAME per TG -> 3 MeOH)
        # More accurately: 1 FAME produced consumes 1 MeOH in each step
        moles_fame_produced = df['FAME'] * total_vol_L # FAME conc * vol
        moles_meoh_consumed = moles_fame_produced # Approximation: 1:1 mole consumption per FAME
        current_moles_meoh = moles_meoh_init - moles_meoh_consumed
        df['MeOH'] = np.maximum(0, current_moles_meoh / total_vol_L) # Ensure non-negative

    elif 'MeOH' not in df.columns: # If FAME also not present, fill with NaN
        df['MeOH'] = np.nan

    return df


def plot_predictions(exp_id, times, true_species, pred_species, conditions):
    """
    Plots the predicted vs true species concentrations for a single experiment.

    Args:
        exp_id (str): Identifier for the experiment.
        times (np.ndarray): Time points.
        true_species (np.ndarray): Ground truth species concentrations (n_times, n_species).
        pred_species (np.ndarray): Predicted species concentrations (n_times, n_species).
        conditions (pd.Series or dict): Experimental conditions for labeling plot.
    """
    n_species = true_species.shape[1]
    if n_species != len(SPECIES):
         print(f"Warning: Mismatch in species count for plotting {exp_id}. Expected {len(SPECIES)}, got {n_species}")
         # Try plotting based on available columns if possible
         plot_species_indices = range(min(n_species, len(SPECIES)))
         plot_species_names = SPECIES[:min(n_species, len(SPECIES))]
    else:
         plot_species_indices = range(n_species)
         plot_species_names = SPECIES

    if len(plot_species_names) == 0:
        print(f"Error: No species to plot for {exp_id}")
        return

    n_rows = int(np.ceil(len(plot_species_names) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4), sharex=True)
    axes = axes.flatten() # Flatten to easily iterate

    condition_str = ", ".join([f"{k}={v:.2f}" for k, v in conditions.items()])
    fig.suptitle(f"Experiment: {exp_id}\nConditions: {condition_str}", fontsize=14)

    for i, species_idx in enumerate(plot_species_indices):
        species_name = plot_species_names[i]
        ax = axes[i]
        ax.plot(times, true_species[:, species_idx], 'o-', label=f'True {species_name}', markersize=4)
        ax.plot(times, pred_species[:, species_idx], 'x--', label=f'Predicted {species_name}', markersize=4)
        ax.set_ylabel("Concentration (mol/L)") # Assumes molar units after processing
        ax.set_title(species_name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Add x-axis label to the bottom plots
    for i in range(len(plot_species_names), len(axes)):
         axes[i].set_xlabel("Time (hours)") # Assumes hours
         axes[i].axis('off') # Hide unused subplots

    # Ensure bottom axes have x-label if they are used
    used_axes_indices = np.arange(len(plot_species_names))
    bottom_row_start_index = (n_rows - 1) * 3
    for i in used_axes_indices:
        if i >= bottom_row_start_index:
             axes[i].set_xlabel("Time (hours)") # Assumes hours


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the figure
    if not os.path.exists(FIGURE_SAVE_DIR):
        os.makedirs(FIGURE_SAVE_DIR)
    filename = f"prediction_{exp_id}.png".replace(" ", "_").replace("/", "_") # Sanitize filename
    filepath = os.path.join(FIGURE_SAVE_DIR, filename)
    try:
        plt.savefig(filepath)
        print(f"Saved prediction plot to {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")
    plt.close(fig) # Close the figure to free memory


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Saves model checkpoint."""
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Model checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath, device):
    """Loads model checkpoint."""
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found: {filepath}")
        return 0, float('inf') # Start from epoch 0, infinite loss

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {filepath} (Epoch {epoch}, Loss {loss:.4f})")
    return epoch + 1, loss # Return next epoch to start from

