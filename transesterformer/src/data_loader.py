# src/data_loader.py
"""
Handles loading, preprocessing, and batching of the kinetic data.

Key fixes
• Added missing denormalize_* imports.
• Recomputed normalization statistics from the actual dataset and pushed
  the new tensors into both constants and utils so every module sees
  consistent values.
• Safer masking when NaNs are present.
• Clearer logging and error handling.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .constants import (
    DATA_PATH,
    TIME_COL,
    EXP_ID_COL,
    CONDITION_COLS,
    SPECIES_COLS,
    N_SPECIES,
    DEVICE,
    BATCH_SIZE,
    USE_NORMALIZATION,
)
from .utils import (
    normalize_conditions,
    normalize_species,
    convert_mass_fraction_to_molar,
    denormalize_conditions,   # newly imported
    denormalize_species,      # newly imported
)

# expose modules so we can update their globals after computing stats
from . import constants as _const
from . import utils as _utils


class BiodieselKineticsDataset(Dataset):
    """PyTorch Dataset for biodiesel kinetic data."""

    def __init__(self, data_path: str = DATA_PATH, device: torch.device = DEVICE):
        self.data_path = data_path
        self.device = device

        self.experiments = self._load_and_process_data()
        if not self.experiments:
            raise ValueError("No valid experiments loaded. Check data path and format.")

        self._calculate_normalization_stats()

    # --------------------------------------------------------------------- #
    # Loading and basic cleaning                                            #
    # --------------------------------------------------------------------- #
    def _load_and_process_data(self):
        try:
            df = pd.read_csv(self.data_path)
            print(f"Loaded data with columns: {df.columns.tolist()}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found at {self.data_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}") from e

        df = df.sort_values(by=[EXP_ID_COL, TIME_COL])

        required = [EXP_ID_COL, TIME_COL] + CONDITION_COLS + SPECIES_COLS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # fill NaNs in conditions
        for col in CONDITION_COLS:
            if df[col].isna().any():
                mean_val = df[col].mean()
                print(f"Filling NaNs in condition '{col}' with mean {mean_val:.2f}")
                df[col].fillna(mean_val, inplace=True)

        print("Assuming species concentrations already molar. Skipping unit conversion.")

        nan_summary = df[SPECIES_COLS].isna().sum()
        if nan_summary.any():
            print("NaNs in species columns will be masked during training:")
            print(nan_summary[nan_summary > 0])

        experiments = []
        grouped = df.groupby(EXP_ID_COL)
        print(f"Processing {len(grouped)} unique experiments...")
        for exp_id, g in grouped:
            if len(g) < 2:
                print(f"Skipping {exp_id}: needs at least two time points.")
                continue

            g = g.sort_values(TIME_COL)

            times = torch.tensor(g[TIME_COL].values, dtype=torch.float32, device=self.device)
            cond_df = g.iloc[[0]][CONDITION_COLS]
            conditions = normalize_conditions(cond_df).squeeze(0).to(self.device)

            raw_species = g[SPECIES_COLS].values
            mask = ~np.isnan(raw_species)
            filled_species = np.nan_to_num(raw_species, nan=0.0)
            species_norm = normalize_species(pd.DataFrame(filled_species, columns=SPECIES_COLS)).to(self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)

            experiments.append(
                {
                    "exp_id": exp_id,
                    "times": times,
                    "conditions": conditions,
                    "species_raw": torch.tensor(raw_species, dtype=torch.float32, device=self.device),
                    "species_norm": species_norm,
                    "initial_conditions": species_norm[0].clone().detach(),
                    "mask": mask_tensor,
                }
            )

        print(f"Successfully processed {len(experiments)} experiments.")
        return experiments

    # --------------------------------------------------------------------- #
    # Normalization statistics                                              #
    # --------------------------------------------------------------------- #
    def _calculate_normalization_stats(self):
        if not USE_NORMALIZATION:
            print("Global normalization disabled — using predefined constants.")
            return

        cond_rows, species_rows = [], []
        for exp in self.experiments:
            cond_rows.append(
                denormalize_conditions(exp["conditions"].unsqueeze(0)).cpu().numpy()
            )

            spec_full = denormalize_species(exp["species_norm"]).reshape(-1, N_SPECIES)
            valid_mask = exp["mask"].cpu().numpy().reshape(-1, N_SPECIES)
            valid_rows = spec_full[valid_mask.all(axis=1)]
            if valid_rows.size:
                species_rows.append(valid_rows)

        if not cond_rows or not species_rows:
            print("Not enough data to recompute normalization stats. Using defaults.")
            return

        cond_arr = np.vstack(cond_rows)
        spec_arr = np.vstack(species_rows)

        cond_means = np.mean(cond_arr, axis=0)
        cond_stds = np.std(cond_arr, axis=0)
        spec_means = np.mean(spec_arr, axis=0)
        spec_stds = np.std(spec_arr, axis=0)

        cond_stds[cond_stds < 1e-8] = 1.0
        spec_stds[spec_stds < 1e-8] = 1.0

        _const.CONDITION_MEANS = torch.tensor(cond_means, dtype=torch.float32, device=self.device)
        _const.CONDITION_STDS = torch.tensor(cond_stds, dtype=torch.float32, device=self.device)
        _const.SPECIES_MEANS = torch.tensor(spec_means, dtype=torch.float32, device=self.device)
        _const.SPECIES_STDS = torch.tensor(spec_stds, dtype=torch.float32, device=self.device)

        _utils.CONDITION_MEANS = _const.CONDITION_MEANS
        _utils.CONDITION_STDS = _const.CONDITION_STDS
        _utils.SPECIES_MEANS = _const.SPECIES_MEANS
        _utils.SPECIES_STDS = _const.SPECIES_STDS

        print("Updated normalization constants from dataset.")
        print("Condition means:", cond_means)
        print("Condition stds :", cond_stds)
        print("Species means  :", spec_means)
        print("Species stds   :", spec_stds)

        # re‑normalize cached tensors
        for exp in self.experiments:
            exp["conditions"] = normalize_conditions(
                pd.DataFrame(
                    [denormalize_conditions(exp["conditions"].unsqueeze(0)).cpu().numpy().squeeze()],
                    columns=CONDITION_COLS,
                )
            ).squeeze(0).to(self.device)

            raw_df = pd.DataFrame(exp["species_raw"].cpu().numpy(), columns=SPECIES_COLS)
            filled = np.nan_to_num(raw_df.values, nan=0.0)
            exp["species_norm"] = normalize_species(pd.DataFrame(filled, columns=SPECIES_COLS)).to(
                self.device
            )
            exp["initial_conditions"] = exp["species_norm"][0].clone().detach()

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, idx):
        return self.experiments[idx]


# ---------------------------------------------------------------------- #
def get_dataloader(
    data_path: str = DATA_PATH,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    device: torch.device = DEVICE,
):
    dataset = BiodieselKineticsDataset(data_path=data_path, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)


if __name__ == "__main__":
    print(f"Loading data on {DEVICE}")
    loader = get_dataloader(batch_size=2, shuffle=False)
    print(f"Dataset size: {len(loader.dataset)} experiments")
