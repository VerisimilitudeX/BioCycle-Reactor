# src/constants.py
"""
Defines constants used throughout the project, including physical constants,
species names, expected data columns, and model hyperparameters.
"""

import torch

# --- Physical/Chemical Constants ---
# Approximate Molecular Weights (g/mol) - Adjust based on the specific oil used!
# Assuming average values for a typical vegetable oil (like soybean or sunflower)
MW_OIL = 875.0  # Average MW of Triglyceride (e.g., Triolein C57H104O6)
MW_MEOH = 32.04 # Methanol
MW_FAME = 296.0 # Average MW of Methyl Ester (e.g., Methyl Oleate C19H36O2)
MW_GLY = 92.09  # Glycerol

# --- Species Names & Indices ---
# Ensure these match the columns in your processed data CSV
SPECIES = ['TG', 'MeOH', 'FAME', 'Gly'] # Order matters!
SPECIES_MAP = {name: i for i, name in enumerate(SPECIES)}
N_SPECIES = len(SPECIES)

# --- Data Columns ---
# Columns expected in the input CSV data/processed/kinetic_data.csv
TIME_COL = 'time'
EXP_ID_COL = 'exp_id'
CONDITION_COLS = ['temperature', 'catalyst_conc'] # Input conditions
SPECIES_COLS = SPECIES # Output species concentrations

# --- Model Hyperparameters ---
# Neural ODE settings
NODE_HIDDEN_DIM = 64   # Hidden dimension of the neural network within the ODE function
NODE_N_LAYERS = 2      # Number of hidden layers in the ODE function's network
NODE_ACTIVATION = torch.nn.Tanh # Activation function

# Condition Encoder settings
ENCODER_HIDDEN_DIM = 32
ENCODER_N_LAYERS = 2
ENCODER_OUTPUT_DIM = 16 # Dimension of the encoded condition vector

# Symbolic Kinetics Parameters (Initial guesses or bounds if needed)
# These might be learned or fixed depending on the model variant
INITIAL_K_CAT_GUESS = 1.0 # Example initial guess for catalytic constants
INITIAL_KM_GUESS = 0.1    # Example initial guess for Michaelis constants

# --- Training Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16      # Number of experiments per batch
N_EPOCHS = 500       # Total training epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5  # L2 regularization
ODE_SOLVER = 'dopri5' # Choose from 'dopri5', 'adams', 'rk4' etc. (torchdiffeq)
ODE_TOLERANCE = 1e-4 # Solver tolerance (rtol=atol=ODE_TOLERANCE)
PRINT_FREQ = 10      # Print loss every N epochs
SAVE_FREQ = 50       # Save model checkpoint every N epochs

# Physics-based Loss Weights (tune these carefully)
LAMBDA_MASS_BALANCE = 0.0 # Weight for mass balance penalty (can be 0 if enforced by structure)
LAMBDA_RATE_REG = 1e-6    # Regularization on the magnitude of neural rate corrections

# --- File Paths ---
DATA_PATH = "data/processed/kinetic_data.csv"
MODEL_SAVE_DIR = "results/model_checkpoint/"
FIGURE_SAVE_DIR = "results/figures/"

# --- Stoichiometry ---
# Represents the net change in moles for each species per mole of reaction progress
# Reactions:
# 1: TG + MeOH <-> DG + FAME
# 2: DG + MeOH <-> MG + FAME
# 3: MG + MeOH <-> Gly + FAME
# Columns: TG, DG, MG, FAME, Gly, MeOH
STOICHIOMETRY_MATRIX = torch.tensor([
    [-1,  1,  0,  1,  0, -1], # Reaction 1
    [ 0, -1,  1,  1,  0, -1], # Reaction 2
    [ 0,  0, -1,  1,  1, -1], # Reaction 3
], dtype=torch.float32, device=DEVICE)
N_REACTIONS = STOICHIOMETRY_MATRIX.shape[0]

# --- Normalization ---
# Define means and stds for conditions and species if using normalization
# These should ideally be calculated from the training data
# Example placeholders (replace with actual values calculated in data_loader):
CONDITION_MEANS = torch.tensor([45.0, 1.0], device=DEVICE) # Temperature, catalyst_conc
CONDITION_STDS = torch.tensor([10.0, 0.5], device=DEVICE)
SPECIES_MEANS = torch.tensor([0.5, 0.5, 0.5, 0.1], device=DEVICE) # TG, MeOH, FAME, Gly (mol/L)
SPECIES_STDS = torch.tensor([0.4, 0.4, 0.4, 0.1], device=DEVICE)
USE_NORMALIZATION = True # Set to False if you don't want normalization

