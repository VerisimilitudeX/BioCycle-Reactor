# run_training.py
"""
Main script to start the training process.
"""
import os
import argparse
from src.train import run_training_pipeline
from src.constants import DATA_PATH, MODEL_SAVE_DIR

def main():
    parser = argparse.ArgumentParser(description="Train Transesterformer Model")
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                        help='Path to the processed kinetic data CSV file.')
    parser.add_argument('--load_model', type=str, default=None, # Example: 'results/model_checkpoint/best_model.pth'
                        help='Path to a checkpoint file to resume training.')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force using CPU even if CUDA is available.')

    args = parser.parse_args()

    # --- Device Setup ---
    if args.force_cpu:
        import torch
        from src import constants
        constants.DEVICE = torch.device("cpu")
        # Need to potentially reload modules if DEVICE was used at import time
        # This is tricky, better to set device early or pass it around.
        # For simplicity, we assume constants.py sets DEVICE correctly based on initial check.
        # If forcing CPU, ensure constants.DEVICE reflects this before model/data loading.
        print("Forcing CPU usage.")
        # Update constants directly (use with caution)
        import src.constants
        src.constants.DEVICE = torch.device("cpu")


    # --- Create result directories if they don't exist ---
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    if not os.path.exists(os.path.join("results", "figures")): # Ensure figure dir exists
         os.makedirs(os.path.join("results", "figures"))


    # --- Run Training ---
    run_training_pipeline(
        data_path=args.data_path,
        model_load_path=args.load_model
    )

if __name__ == "__main__":
    main()
