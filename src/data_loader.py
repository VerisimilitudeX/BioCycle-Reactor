# src/data_loader.py
"""
Handles loading, preprocessing, and batching of the kinetic data.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .constants import (
    DATA_PATH, TIME_COL, EXP_ID_COL, CONDITION_COLS, SPECIES_COLS, N_SPECIES,
    DEVICE, BATCH_SIZE, USE_NORMALIZATION
)
from .utils import (
    normalize_conditions, normalize_species, convert_mass_fraction_to_molar,
    denormalize_conditions, denormalize_species
)

class BiodieselKineticsDataset(Dataset):
    """
    PyTorch Dataset class for loading and serving biodiesel kinetic data.
    Each item corresponds to a single experiment's time series.
    """
    def __init__(self, data_path=DATA_PATH, device=DEVICE):
        self.data_path = data_path
        self.device = device
        self.experiments = self._load_and_process_data()

        if not self.experiments:
            raise ValueError("No valid experiments loaded. Check data path and format.")

        # Calculate normalization stats if needed (and not predefined)
        # This is simplified; ideally done only on training split
        self._calculate_normalization_stats()


    def _load_and_process_data(self):
        """Loads data from CSV, processes it, and groups by experiment."""
        try:
            df = pd.read_csv(self.data_path, sep=",", engine="python")
            df = df.rename(columns={
                'oil': 'TG',
                'methanol': 'MeOH',
                'biodiesel': 'FAME',
                'glycerol': 'Gly'
            })
            print(f"Loaded data with columns: {df.columns.tolist()}")
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            return []
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

        # --- Data Cleaning & Preprocessing ---
        # 1. Handle Missing Values (Example: forward fill within experiment, then drop if still NaN)
        df = df.sort_values(by=[EXP_ID_COL, TIME_COL])
        # Check for essential columns
        required_cols = [EXP_ID_COL, TIME_COL] + CONDITION_COLS + SPECIES_COLS
        missing_req_cols = [col for col in required_cols if col not in df.columns and col not in SPECIES_COLS] # Allow missing species initially
        if missing_req_cols:
             print(f"Error: Missing required columns: {missing_req_cols}")
             # Attempt to continue if only optional conditions are missing
             if any(c in CONDITION_COLS for c in missing_req_cols):
                 print("Warning: Missing some condition columns. Will fill with NaN.")
                 for col in missing_req_cols:
                     if col in CONDITION_COLS: df[col] = np.nan
             else: # If essential ID/Time or mandatory conditions missing
                 return []


        # Fill missing conditions with mean or 0 (or handle more sophisticatedly)
        for col in CONDITION_COLS:
            if col in df.columns:
                if df[col].isnull().any():
                    fill_value = df[col].mean() if df[col].notna().any() else 0
                    print(f"Warning: Filling NaNs in condition '{col}' with {fill_value:.2f}")
                    df[col].fillna(fill_value, inplace=True)
            else:
                print(f"Warning: Condition column '{col}' not found. Creating with NaNs.")
                df[col] = np.nan # Create if missing, fill later if needed


        # 2. Unit Conversion (Example: Convert mass % to mol/L if needed)
        # This requires knowledge of initial conditions (e.g., initial oil mass)
        # Placeholder: Assume data is ALREADY in mol/L or skip conversion
        # If you need conversion, implement logic here, potentially grouping by experiment
        # Example call (requires 'initial_oil_mass_g' potentially mapped from exp_id):
        # df = df.groupby(EXP_ID_COL).apply(
        #      lambda x: convert_mass_fraction_to_molar(x, initial_oil_mass_g=100.0) # Adjust mass
        # ).reset_index(drop=True)
        print("Skipping unit conversion. Assuming data is in molar units (mol/L).")
        # Ensure all target species columns exist, fill with NaN if missing
        for col in SPECIES_COLS:
            if col not in df.columns:
                print(f"Warning: Species column '{col}' not found. Creating with NaNs.")
                df[col] = np.nan

        # 3. Handle missing species data (critical!)
        # Option 1: Drop experiments with any NaN in species after potential ffill
        # Option 2: Impute (less ideal for time series)
        # Option 3: Mask loss for NaN values during training
        # Here, we'll use Option 3 (masking) - requires handling in training loop
        # Check for NaNs *before* normalization
        nan_counts = df[SPECIES_COLS].isnull().sum()
        if nan_counts.sum() > 0:
            print(f"Warning: Found NaNs in species columns:\n{nan_counts[nan_counts > 0]}")
            print("These time points will be masked during loss calculation.")


        # --- Group by Experiment ---
        grouped = df.groupby(EXP_ID_COL)
        experiments = []
        print(f"Processing {len(grouped)} unique experiments...")

        for exp_id, group in grouped:
            if len(group) < 2: # Need at least two time points for ODE
                print(f"Skipping experiment {exp_id}: only {len(group)} time point(s).")
                continue

            # Ensure time is sorted
            group = group.sort_values(by=TIME_COL)

            times = torch.tensor(group[TIME_COL].values, dtype=torch.float32, device=self.device)
            # Extract conditions (use first row's conditions for the whole experiment)
            conditions_df = group.iloc[[0]] # Get first row as DataFrame
            conditions = normalize_conditions(conditions_df) if USE_NORMALIZATION else torch.tensor(conditions_df[CONDITION_COLS].values, dtype=torch.float32)


            # Extract species concentrations and handle potential NaNs
            species_raw = group[SPECIES_COLS].values
            species_mask = ~np.isnan(species_raw) # True where data is valid
            species_filled = np.nan_to_num(species_raw, nan=0.0) # Replace NaN with 0 for tensor creation

            species = normalize_species(pd.DataFrame(species_filled, columns=SPECIES_COLS)) if USE_NORMALIZATION else torch.tensor(species_filled, dtype=torch.float32)
            mask_tensor = torch.tensor(species_mask, dtype=torch.bool, device=self.device)


            # Check shapes
            if species.shape[1] != N_SPECIES:
                 print(f"Error: Experiment {exp_id} has incorrect species dimension: {species.shape[1]}, expected {N_SPECIES}")
                 continue
            if conditions.shape[1] != len(CONDITION_COLS):
                 print(f"Error: Experiment {exp_id} has incorrect condition dimension: {conditions.shape[1]}, expected {len(CONDITION_COLS)}")
                 continue


            # Store initial condition separately
            initial_conditions = species[0, :].clone().detach().to(self.device) # Normalized or raw C(t=0)

            experiments.append({
                'exp_id': exp_id,
                'times': times.to(self.device),             # Time points (vector)
                'conditions': conditions.squeeze(0).to(self.device), # Conditions (vector)
                'species_raw': torch.tensor(species_raw, dtype=torch.float32, device=self.device), # Raw data with NaNs
                'species_norm': species.to(self.device),    # Normalized/raw data (tensor, NaNs filled with 0)
                'initial_conditions': initial_conditions, # C(t=0)
                'mask': mask_tensor.to(self.device)         # Mask for valid data points
            })

        print(f"Successfully processed {len(experiments)} experiments.")
        return experiments

    def _calculate_normalization_stats(self):
      """Calculates and updates normalization constants if USE_NORMALIZATION is True."""
      if not USE_NORMALIZATION:
          print("Normalization disabled.")
          return

      all_conditions = []
      all_species = []
      for exp in self.experiments:
          # Denormalize first if they were already normalized with placeholders
          cond_numpy = denormalize_conditions(exp['conditions'].unsqueeze(0)).cpu().numpy()
          spec_numpy = denormalize_species(exp['species_norm']).cpu().numpy() # Denormalize filled data

          # Use the mask to only consider valid data points for stats
          valid_spec_numpy = spec_numpy[exp['mask'].cpu().numpy()]
          # Reshape needed if mask is applied per element
          if valid_spec_numpy.size == exp['mask'].sum().item() * N_SPECIES:
               valid_spec_numpy = valid_spec_numpy.reshape(-1, N_SPECIES)
          else: # Fallback if reshape fails (e.g. all NaNs in a row)
               valid_spec_numpy = spec_numpy[exp['mask'].all(axis=1).cpu().numpy()] # Use only rows with all valid data


          all_conditions.append(cond_numpy)
          if valid_spec_numpy.shape[0] > 0: # Only append if valid species data exists
              all_species.append(valid_spec_numpy)

      if not all_conditions or not all_species:
          print("Warning: Not enough valid data to calculate normalization statistics.")
          # Keep predefined constants or disable normalization
          # global USE_NORMALIZATION # Commented out: Avoid modifying global constants directly
          # USE_NORMALIZATION = False
          print("Keeping predefined normalization constants or disabling normalization.")
          return

      conditions_all_np = np.concatenate(all_conditions, axis=0)
      species_all_np = np.concatenate(all_species, axis=0)

      # Calculate means and stds
      cond_means = np.nanmean(conditions_all_np, axis=0)
      cond_stds = np.nanstd(conditions_all_np, axis=0)
      spec_means = np.nanmean(species_all_np, axis=0)
      spec_stds = np.nanstd(species_all_np, axis=0)

      # Handle zero std deviation (replace with 1 or small epsilon)
      cond_stds[cond_stds < 1e-8] = 1.0
      spec_stds[spec_stds < 1e-8] = 1.0

      # Update constants (This is generally bad practice to modify imported constants,
      # better to pass them around or use a config object. Doing it here for simplicity.)
      global CONDITION_MEANS, CONDITION_STDS, SPECIES_MEANS, SPECIES_STDS
      CONDITION_MEANS = torch.tensor(cond_means, dtype=torch.float32, device=self.device)
      CONDITION_STDS = torch.tensor(cond_stds, dtype=torch.float32, device=self.device)
      SPECIES_MEANS = torch.tensor(spec_means, dtype=torch.float32, device=self.device)
      SPECIES_STDS = torch.tensor(spec_stds, dtype=torch.float32, device=self.device)

      print("Updated normalization constants based on loaded data:")
      print("Condition Means:", CONDITION_MEANS.cpu().numpy())
      print("Condition Stds:", CONDITION_STDS.cpu().numpy())
      print("Species Means:", SPECIES_MEANS.cpu().numpy())
      print("Species Stds:", SPECIES_STDS.cpu().numpy())

      # Re-normalize the data in self.experiments with the calculated stats
      print("Re-normalizing loaded experiment data...")
      for i in range(len(self.experiments)):
          exp_data = self.experiments[i]
          # Conditions (use the raw conditions stored before initial normalization)
          raw_cond_df = pd.DataFrame([denormalize_conditions(exp_data['conditions'].unsqueeze(0)).cpu().numpy().squeeze()], columns=CONDITION_COLS)
          self.experiments[i]['conditions'] = normalize_conditions(raw_cond_df).squeeze(0).to(self.device)

          # Species (use the raw species data before initial normalization)
          raw_spec_df = pd.DataFrame(exp_data['species_raw'].cpu().numpy(), columns=SPECIES_COLS)
          species_filled = np.nan_to_num(raw_spec_df.values, nan=0.0)
          self.experiments[i]['species_norm'] = normalize_species(pd.DataFrame(species_filled, columns=SPECIES_COLS)).to(self.device)

          # Update initial conditions as well
          self.experiments[i]['initial_conditions'] = self.experiments[i]['species_norm'][0, :].clone().detach().to(self.device)
      print("Re-normalization complete.")


    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        return self.experiments[idx]

def get_dataloader(data_path=DATA_PATH, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE):
    """Creates and returns a DataLoader."""
    dataset = BiodieselKineticsDataset(data_path=data_path, device=device)
    # Collate function to handle variable length sequences if needed (not strictly necessary here)
    # DataLoader will automatically stack tensors if they have the same size,
    # but experiments have different numbers of time points.
    # We process experiments individually in the training loop, so standard loader is fine.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x) # Return list of dicts
    return dataloader

if __name__ == '__main__':
    # Example usage: Load data and print info about the first batch
    print(f"Loading data using device: {DEVICE}")
    dataloader = get_dataloader(batch_size=4, shuffle=False)
    print(f"DataLoader created. Number of experiments: {len(dataloader.dataset)}")

    first_batch = next(iter(dataloader))
    print(f"\nFirst batch contains {len(first_batch)} experiments.")

    if first_batch:
        first_exp = first_batch[0]
        print("\nDetails of the first experiment in the batch:")
        print(f"  Experiment ID: {first_exp['exp_id']}")
        print(f"  Number of time points: {len(first_exp['times'])}")
        print(f"  Times shape: {first_exp['times'].shape}")
        print(f"  Conditions shape: {first_exp['conditions'].shape}")
        print(f"  Conditions (normalized): {first_exp['conditions']}")
        print(f"  Initial Conditions (normalized): {first_exp['initial_conditions']}")
        print(f"  Species Norm shape: {first_exp['species_norm'].shape}")
        print(f"  Mask shape: {first_exp['mask'].shape}")
        print(f"  Number of valid data points: {first_exp['mask'].sum().item()}")

        # Denormalize conditions and species for inspection
        if USE_NORMALIZATION:
             from .utils import denormalize_conditions, denormalize_species
             denorm_cond = denormalize_conditions(first_exp['conditions'].unsqueeze(0))
             denorm_spec_init = denormalize_species(first_exp['initial_conditions'].unsqueeze(0))
             print(f"  Conditions (denormalized): {denorm_cond.cpu().numpy().squeeze()}")
             print(f"  Initial Conditions (denormalized): {denorm_spec_init.squeeze()}") # Already numpy
