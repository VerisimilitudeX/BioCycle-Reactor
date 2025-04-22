# src/model.py
"""
Defines the Physics‑Guided Neural ODE model (“Transesterformer”),
with robust per‑sample integration and proper tensor shape handling.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint            # use standard odeint solver

from .constants import (
    N_SPECIES, N_REACTIONS, CONDITION_COLS,
    NODE_HIDDEN_DIM, NODE_N_LAYERS, NODE_ACTIVATION,
    ENCODER_HIDDEN_DIM, ENCODER_N_LAYERS, ENCODER_OUTPUT_DIM,
    STOICHIOMETRY_MATRIX, DEVICE, SPECIES_MAP, LAMBDA_RATE_REG
)

# -------------------------------------------------------------------------
class ConditionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------------------
class SymbolicKinetics(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        # Define layers to predict kinetic parameters based on encoded conditions
        self.vmax_net  = nn.Linear(enc_dim, N_REACTIONS) # Predicts Vmax for each reaction
        self.km_tg_net = nn.Linear(enc_dim, 1)           # Predicts Km for TG in reaction 1
        self.km_dg_net = nn.Linear(enc_dim, 1)           # Predicts Km for DG in reaction 2
        self.km_mg_net = nn.Linear(enc_dim, 1)           # Predicts Km for MG in reaction 3
        self.km_meoh   = nn.Linear(enc_dim, N_REACTIONS) # Predicts Km for MeOH in each reaction
        self.softplus  = nn.Softplus()                   # Ensures kinetic parameters are positive

    def forward(self, species, enc):
        """
        Calculates reaction rates based on Michaelis-Menten kinetics.
        Args:
            species (Tensor): Species concentrations, shape (batch, n_species).
            enc (Tensor): Encoded conditions, shape (batch, enc_dim).
        Returns:
            Tensor: Reaction rates, shape (batch, n_reactions).
        """
        # Predict kinetic parameters from encoded conditions
        # Use softplus to ensure positivity
        vmax  = self.softplus(self.vmax_net(enc))    # (batch, n_reactions)
        km_tg = self.softplus(self.km_tg_net(enc))   # (batch, 1)
        km_dg = self.softplus(self.km_dg_net(enc))   # (batch, 1)
        km_mg = self.softplus(self.km_mg_net(enc))   # (batch, 1)
        km_me = self.softplus(self.km_meoh(enc))     # (batch, n_reactions)

        # Extract species concentrations, ensuring non-negativity with relu
        # Indexing assumes species is (batch, n_species)
        tg = torch.relu(species[:, SPECIES_MAP['TG']])   # (batch,)
        dg = torch.relu(species[:, SPECIES_MAP['DG']])   # (batch,)
        mg = torch.relu(species[:, SPECIES_MAP['MG']])   # (batch,)
        me = torch.relu(species[:, SPECIES_MAP['MeOH']]) # (batch,)
        eps = 1e-8 # Small epsilon to prevent division by zero in kinetics

        # Calculate rates for each reaction using Michaelis-Menten like terms
        # Unsqueeze substrate concentrations to allow broadcasting with Km tensors
        # r = Vmax * [S1]/(Km1 + [S1]) * [S2]/(Km2 + [S2])
        r1 = vmax[:,0] * tg / (km_tg.squeeze(-1) + tg + eps) * me / (km_me[:,0] + me + eps)
        r2 = vmax[:,1] * dg / (km_dg.squeeze(-1) + dg + eps) * me / (km_me[:,1] + me + eps)
        r3 = vmax[:,2] * mg / (km_mg.squeeze(-1) + mg + eps) * me / (km_me[:,2] + me + eps)

        # Stack rates into a single tensor (batch, n_reactions)
        rates = torch.stack([r1, r2, r3], dim=1)
        # Ensure rates are non-negative
        return torch.relu(rates)

# -------------------------------------------------------------------------
class NeuralAugmentation(nn.Module):
    def __init__(self, n_species, enc_dim, hidden_dim, n_layers, activation):
        super().__init__()
        # Define the neural network structure
        layers = [nn.Linear(n_species + enc_dim, hidden_dim), activation()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, N_REACTIONS)) # Output dimension matches number of reactions
        self.net = nn.Sequential(*layers)

    def forward(self, species, enc):
        """
        Calculates the neural correction/augmentation to reaction rates.
        Args:
            species (Tensor): Species concentrations, shape (batch, n_species).
            enc (Tensor): Encoded conditions, shape (batch, enc_dim).
        Returns:
            Tensor: Neural rate adjustments, shape (batch, n_reactions).
        """
        # Concatenate species and encoded conditions as input to the network
        nn_input = torch.cat([species, enc], dim=1)
        return self.net(nn_input)

# -------------------------------------------------------------------------
class TransesterformerODE(nn.Module):
    """
    The ODE function dy/dt = f(t, y, conditions) for the Transesterformer model.
    It combines symbolic kinetics with a neural augmentation term.
    Designed to be called by an ODE solver like torchdiffeq.odeint.
    """
    def __init__(self, enc_dim, hidden_dim, n_layers, activation):
        super().__init__()
        self.symbolic = SymbolicKinetics(enc_dim)
        self.neural   = NeuralAugmentation(N_SPECIES, enc_dim, hidden_dim, n_layers, activation)
        # Ensure Stoichiometry matrix is on the correct device
        self.S        = STOICHIOMETRY_MATRIX.to(DEVICE)
        self._enc     = None # To store the current encoded conditions
        # To store the regularization loss from the neural component
        self.latest_neural_reg_loss = torch.tensor(0.0, device=DEVICE)

    def set_conditions(self, enc):
        """
        Store the encoded conditions for the current ODE solve.
        Ensures enc is at least 2D: (1, enc_dim).
        """
        if enc.dim() == 1:
            # Unsqueeze if a single condition vector is passed
            self._enc = enc.unsqueeze(0).to(DEVICE)
        else:
            self._enc = enc.to(DEVICE)

    def forward(self, t, species):
        """
        Computes the time derivative of species concentrations (dC/dt).
        Args:
            t (Tensor): Current time (scalar). Usually ignored in autonomous ODEs.
            species (Tensor): Current species concentrations.
                              Shape can be (n_species,) or (batch, n_species)
                              depending on how odeint calls it.
        Returns:
            Tensor: Time derivatives (dC/dt), same shape as input `species`.
        """
        enc = self._enc
        if enc is None:
            raise RuntimeError("Encoded conditions not set. Call set_conditions() before odeint.")

        # --- Shape Handling ---
        # Store original dimension to return the correct shape expected by odeint
        original_dim = species.dim()
        if original_dim == 1:
            # Unsqueeze to (1, n_species) if input is 1D
            species = species.unsqueeze(0)

        # --- Ensure Condition Encoding Matches Batch Dimension ---
        # This handles cases where odeint might internally batch, or if
        # this function is called directly with batched data.
        if enc.size(0) != species.size(0):
            if enc.size(0) == 1:
                # Broadcast conditions if only one set was provided for a batch
                enc = enc.expand(species.size(0), -1)
            else:
                # This indicates a mismatch that shouldn't happen with the current
                # training loop structure (calling odeint per sample).
                raise RuntimeError(
                    f"Batch size mismatch in ODE function: "
                    f"species batch={species.size(0)}, "
                    f"conditions batch={enc.size(0)}"
                )
        # Ensure species is on the correct device (might be created by odeint on CPU)
        species = species.to(DEVICE)

        # --- Rate Calculation ---
        # Compute rates from symbolic and neural components
        # Both expect 2D input: (batch, n_species) and (batch, enc_dim)
        sym_rates = self.symbolic(species, enc) # (batch, n_reactions)
        neu_rates = self.neural(species, enc)   # (batch, n_reactions)

        # Store the regularization loss (L2 norm of neural rates)
        # Average over the batch dimension if batch > 1
        self.latest_neural_reg_loss = LAMBDA_RATE_REG * torch.mean(neu_rates ** 2)

        # Combine rates (ensure positivity)
        total_rates = torch.relu(sym_rates + neu_rates) # (batch, n_reactions)

        # --- Calculate dC/dt ---
        # Matrix multiplication: (batch, n_reactions) @ (n_reactions, n_species)
        dCdt = total_rates @ self.S # (batch, n_species)

        # --- Return Correct Shape ---
        # odeint expects the returned derivative to have the same shape as the input state
        if original_dim == 1:
            # Squeeze back to (n_species,) if input was 1D
            return dCdt.squeeze(0)
        else:
            # Return (batch, n_species) if input was 2D
            return dCdt

# -------------------------------------------------------------------------
class Transesterformer(nn.Module):
    """
    Main Physics-Guided Neural ODE model. Encodes conditions and uses
    TransesterformerODE with an ODE solver to predict species evolution.
    """
    def __init__(
        self,
        n_conditions=len(CONDITION_COLS),
        encoder_hidden_dim=ENCODER_HIDDEN_DIM,
        encoder_output_dim=ENCODER_OUTPUT_DIM,
        encoder_n_layers=ENCODER_N_LAYERS,
        node_hidden_dim=NODE_HIDDEN_DIM,
        node_n_layers=NODE_N_LAYERS,
        node_activation=NODE_ACTIVATION,
        ode_solver="dopri5", # Default solver
        ode_options=None,    # Dictionary for solver options like tolerances
    ):
        super().__init__()
        # Condition encoder sub-module
        self.encoder     = ConditionEncoder(n_conditions, encoder_hidden_dim,
                                            encoder_output_dim, encoder_n_layers)
        # ODE function sub-module
        self.ode_func    = TransesterformerODE(encoder_output_dim,
                                               node_hidden_dim,
                                               node_n_layers,
                                               node_activation)
        self.ode_solver  = ode_solver
        # Store ODE solver options, ensuring defaults are set if none provided
        self.ode_options = ode_options.copy() if ode_options else {}
        # Initialize regularization loss storage
        self.neural_reg_loss = torch.tensor(0.0, device=DEVICE)

    def forward(self, initial_conditions, times, conditions):
        """
        Performs the forward pass: encodes conditions, solves the ODE for each
        sample in the batch, and returns the predicted species concentrations over time.

        Args:
            initial_conditions (Tensor): Initial species concentrations (batch, n_species).
            times (Tensor): Time points for prediction (n_times,). Should start at t=0.
            conditions (Tensor): Experimental conditions (batch, n_conditions).

        Returns:
            Tensor: Predicted species concentrations (batch, n_times, n_species).
        """
        batch_size = initial_conditions.size(0)
        out_list = [] # List to store solutions for each batch item

        # --- Loop through batch samples ---
        # NOTE: This processes each sample individually. For larger datasets,
        #       calling odeint once with the full batch might be more efficient,
        #       but requires the ODE function to handle batches correctly.
        #       The current TransesterformerODE is designed to handle this,
        #       but this loop structure keeps the logic simpler for now.
        for i in range(batch_size):
            ic = initial_conditions[i]         # Get initial condition (n_species,)
            cond_i = conditions[i].unsqueeze(0) # Get condition (1, n_conditions)

            # Encode the condition for this sample
            enc_i = self.encoder(cond_i)       # (1, enc_dim)

            # Set the encoded condition in the ODE function instance
            # This makes 'enc_i' available within self.ode_func.forward
            self.ode_func.set_conditions(enc_i)

            # Prepare solver options (e.g., tolerances)
            opts = self.ode_options.copy()
            rtol = opts.pop("rtol", 1e-4) # Relative tolerance
            atol = opts.pop("atol", 1e-4) # Absolute tolerance

            # --- Solve the ODE ---
            sol = odeint(
                self.ode_func,        # The ODE function dC/dt = f(t, C)
                ic,                   # Initial state C(t=0), shape (n_species,)
                times,                # Time points to evaluate at, shape (n_times,)
                rtol=rtol,            # Relative tolerance
                atol=atol,            # Absolute tolerance
                method=self.ode_solver,# Name of the solver algorithm
                options=opts,         # Other solver options
            ) # Output shape: (n_times, n_species)

            # Add batch dimension and append to list
            out_list.append(sol.unsqueeze(0))  # (1, n_times, n_species)

        # Concatenate results from all samples into a single batch tensor
        result = torch.cat(out_list, dim=0)    # (batch, n_times, n_species)

        # Store the regularization loss from the last ODE solve in the loop
        # Note: This only stores the loss from the *last* sample.
        # A more accurate approach would be to average the loss across the batch.
        # This could be done by accumulating loss inside the loop or modifying
        # the training loop to handle loss calculation differently.
        # For now, we keep the existing behavior.
        self.neural_reg_loss = self.ode_func.latest_neural_reg_loss
        return result

# -------------------------------------------------------------------------
# Example Usage / Testing Block
if __name__ == "__main__":
    print(f"Testing Transesterformer on {DEVICE}")
    # Instantiate the model with default parameters
    model = Transesterformer(ode_options={"rtol":1e-3,"atol":1e-3}).to(DEVICE)

    # Create dummy input data
    B, T = 2, 10 # Batch size = 2, Time points = 10
    # Initial conditions (Batch, Species)
    ic = torch.rand(B, N_SPECIES, device=DEVICE)
    # Time vector (Time points,)
    tv = torch.linspace(0, 5, T, device=DEVICE)
    # Conditions (Batch, Conditions)
    conds = torch.rand(B, len(CONDITION_COLS), device=DEVICE)

    # Perform a forward pass without gradient calculation
    with torch.no_grad():
        out = model(ic, tv, conds)

    # Print the output shape
    print("Input IC shape:", ic.shape)
    print("Input times shape:", tv.shape)
    print("Input conditions shape:", conds.shape)
    print("Output shape:", out.shape) # Expected: (B, T, N_SPECIES) -> (2, 10, 6)
    print("Neural Reg Loss (last sample):", model.neural_reg_loss.item())

