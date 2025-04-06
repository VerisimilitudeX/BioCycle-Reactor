# src/model.py
"""
Defines the Physics-Guided Neural ODE model ("Transesterformer").
Includes the condition encoder, the symbolic kinetics part, the neural augmentation part,
and the overall ODE function.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint # Use adjoint method for memory efficiency
# from torchdiffeq import odeint # Use standard odeint if adjoint causes issues

from .constants import (
    N_SPECIES, N_REACTIONS, CONDITION_COLS, NODE_HIDDEN_DIM, NODE_N_LAYERS, NODE_ACTIVATION,
    ENCODER_HIDDEN_DIM, ENCODER_N_LAYERS, ENCODER_OUTPUT_DIM, STOICHIOMETRY_MATRIX,
    DEVICE, SPECIES_MAP, MW_MEOH, MW_OIL, MW_FAME, MW_GLY, LAMBDA_RATE_REG
)

class ConditionEncoder(nn.Module):
    """Encodes high-dimensional condition vector into a lower-dimensional latent space."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU()) # Use ReLU for encoder
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, conditions):
        """
        Args:
            conditions (torch.Tensor): Shape (batch_size, n_conditions)
        Returns:
            torch.Tensor: Shape (batch_size, output_dim)
        """
        return self.net(conditions)

class SymbolicKinetics(nn.Module):
    """
    Implements the symbolic part of the reaction kinetics (e.g., Ping-Pong Bi-Bi).
    Kinetic parameters can be modulated by the encoded conditions.
    """
    def __init__(self, encoded_condition_dim):
        super().__init__()
        # Example: Learnable base parameters (these could be fixed if known)
        # We make Vmax and Km dependent on conditions via small networks
        self.vmax_net = nn.Linear(encoded_condition_dim, N_REACTIONS) # Predict Vmax for each reaction
        self.km_tg_net = nn.Linear(encoded_condition_dim, 1) # Km for TG (assuming same for R1)
        self.km_dg_net = nn.Linear(encoded_condition_dim, 1) # Km for DG (assuming same for R2)
        self.km_mg_net = nn.Linear(encoded_condition_dim, 1) # Km for MG (assuming same for R3)
        self.km_meoh_net = nn.Linear(encoded_condition_dim, N_REACTIONS) # Km for MeOH for each reaction
        self.k_inhibition_meoh_net = nn.Linear(encoded_condition_dim, N_REACTIONS) # Inhibition constant for MeOH

        # Activation to ensure positivity of parameters
        self.softplus = nn.Softplus()

    def forward(self, species, encoded_conditions):
        """
        Calculates reaction rates based on symbolic kinetics.
        Args:
            species (torch.Tensor): Current species concentrations (batch_size, n_species)
            encoded_conditions (torch.Tensor): Encoded condition vector (batch_size, encoded_dim)
        Returns:
            torch.Tensor: Reaction rates (batch_size, n_reactions)
        """
        # Predict condition-dependent parameters
        # Use softplus to ensure positivity
        vmax = self.softplus(self.vmax_net(encoded_conditions)) # (batch_size, n_reactions)
        km_tg = self.softplus(self.km_tg_net(encoded_conditions)) # (batch_size, 1)
        km_dg = self.softplus(self.km_dg_net(encoded_conditions)) # (batch_size, 1)
        km_mg = self.softplus(self.km_mg_net(encoded_conditions)) # (batch_size, 1)
        km_meoh = self.softplus(self.km_meoh_net(encoded_conditions)) # (batch_size, n_reactions)
        k_inhibition_meoh = self.softplus(self.k_inhibition_meoh_net(encoded_conditions)) # (batch_size, n_reactions)

        # Extract individual species concentrations for clarity
        # Ensure species are non-negative before using in denominators
        tg = torch.relu(species[:, SPECIES_MAP['TG']])
        dg = torch.relu(species[:, SPECIES_MAP['DG']])
        mg = torch.relu(species[:, SPECIES_MAP['MG']])
        meoh = torch.relu(species[:, SPECIES_MAP['MeOH']])
        # FAME and Gly are products, don't typically inhibit in simple models

        # --- Simplified Ping-Pong Bi-Bi Kinetics with Methanol Inhibition ---
        # Rate = Vmax * [A] * [B] / (KmB*[A] + KmA*[B] + [A]*[B] * (1 + [I]/Ki))
        # This is a simplified representation. A full Ping-Pong model is more complex.
        # We'll use a Michaelis-Menten like form for each step for simplicity here.
        # Rate_i = Vmax_i * (Substrate1 / (Km1 + Substrate1)) * (Substrate2 / (Km2 + Substrate2)) * InhibitionTerm

        epsilon = 1e-8 # Small value to prevent division by zero

        # Reaction 1: TG + MeOH -> DG + FAME
        denom1 = (km_tg + tg) * (km_meoh[:, 0] + meoh) * (1 + meoh / (k_inhibition_meoh[:, 0] + epsilon))
        rate1 = vmax[:, 0] * (tg / (km_tg + tg + epsilon)) * (meoh / (km_meoh[:, 0] + meoh + epsilon))
        # Simpler alternative: rate1 = vmax[:, 0] * tg * meoh / (denom1 + epsilon) # Check literature for exact form

        # Reaction 2: DG + MeOH -> MG + FAME
        denom2 = (km_dg + dg) * (km_meoh[:, 1] + meoh) * (1 + meoh / (k_inhibition_meoh[:, 1] + epsilon))
        rate2 = vmax[:, 1] * (dg / (km_dg + dg + epsilon)) * (meoh / (km_meoh[:, 1] + meoh + epsilon))

        # Reaction 3: MG + MeOH -> Gly + FAME
        denom3 = (km_mg + mg) * (km_meoh[:, 2] + meoh) * (1 + meoh / (k_inhibition_meoh[:, 2] + epsilon))
        rate3 = vmax[:, 2] * (mg / (km_mg + mg + epsilon)) * (meoh / (km_meoh[:, 2] + meoh + epsilon))

        # Combine rates - ensure shape is (batch_size, n_reactions)
        rates = torch.stack([rate1, rate2, rate3], dim=1)

        # Ensure rates are non-negative
        rates = torch.relu(rates)

        return rates


class NeuralAugmentation(nn.Module):
    """
    Learns residual dynamics (corrections) not captured by the symbolic part.
    Uses a simple MLP.
    """
    def __init__(self, n_species, encoded_condition_dim, hidden_dim, n_layers, activation):
        super().__init__()
        input_dim = n_species + encoded_condition_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        # Output dimension should match the number of reactions or species,
        # depending on how corrections are applied. Here, assume corrections per reaction.
        layers.append(nn.Linear(hidden_dim, N_REACTIONS))
        self.net = nn.Sequential(*layers)

    def forward(self, species, encoded_conditions):
        """
        Args:
            species (torch.Tensor): Current species concentrations (batch_size, n_species)
            encoded_conditions (torch.Tensor): Encoded condition vector (batch_size, encoded_dim)
        Returns:
            torch.Tensor: Rate corrections (batch_size, n_reactions)
        """
        # Concatenate species and conditions as input
        net_input = torch.cat([species, encoded_conditions], dim=1)
        corrections = self.net(net_input)
        return corrections


class TransesterformerODE(nn.Module):
    """
    The combined ODE function dC/dt = f(C, t, conditions).
    Uses the symbolic kinetics and neural augmentation.
    """
    def __init__(self, encoded_condition_dim, node_hidden_dim, node_n_layers, node_activation):
        super().__init__()
        self.encoded_condition_dim = encoded_condition_dim
        self.symbolic_kinetics = SymbolicKinetics(encoded_condition_dim)
        self.neural_augmentation = NeuralAugmentation(
            N_SPECIES, encoded_condition_dim, node_hidden_dim, node_n_layers, node_activation
        )
        # Ensure stoichiometry matrix is on the correct device
        self.stoichiometry = STOICHIOMETRY_MATRIX.to(DEVICE)

        # Store encoded conditions - these are set externally per batch/experiment
        self._encoded_conditions = None
        self._batch_size = None

    def set_conditions(self, encoded_conditions):
        """Stores the encoded conditions for the current batch/integration."""
        self._encoded_conditions = encoded_conditions
        self._batch_size = encoded_conditions.shape[0]


    def forward(self, t, species):
        """
        Calculates dC/dt.
        Args:
            t (torch.Tensor): Current time (scalar, ignored by autonomous ODE).
            species (torch.Tensor): Current species concentrations (batch_size, n_species).
                                     Shape might be just (n_species,) during integration per sample.
        Returns:
            torch.Tensor: Time derivatives dC/dt (batch_size, n_species).
        """
        # Handle potential shape difference during integration
        is_batch = species.dim() == 2
        current_batch_size = species.shape[0] if is_batch else 1

        if self._encoded_conditions is None:
            raise RuntimeError("Encoded conditions not set before calling ODE function.")

        # Ensure encoded conditions match the batch size being processed by the solver
        if is_batch:
            if current_batch_size != self._batch_size:
                 # This might happen if the solver calls with a different batch structure?
                 # Or if called outside the main training loop without setting conditions.
                 # For simplicity, assume conditions are correctly set per batch.
                 # If issues arise, may need to pass conditions explicitly or handle slicing.
                 print(f"Warning: Batch size mismatch in ODE forward. Expected {self._batch_size}, got {current_batch_size}.")
                 # Use the first condition if sizes mismatch (potential issue)
                 encoded_conditions_batch = self._encoded_conditions[0:1].expand(current_batch_size, -1)

            else:
                 encoded_conditions_batch = self._encoded_conditions
        else:
            # If input is single sample (n_species,), use the corresponding condition
            # This assumes the solver integrates sample by sample or we handle batching outside
            # For simplicity, assume conditions are batched correctly matching species input
            if self._batch_size == 1:
                 encoded_conditions_batch = self._encoded_conditions
                 species = species.unsqueeze(0) # Add batch dim
            else:
                 # This case is tricky - which condition corresponds to this single species vector?
                 # Assumes called within a loop where conditions are correctly sliced/indexed.
                 # Fallback: Use the first condition (likely wrong if batch > 1)
                 # encoded_conditions_batch = self._encoded_conditions[0:1]
                 # species = species.unsqueeze(0) # Add batch dim
                 # Better approach: Ensure ODE is always called with batched species & conditions
                 raise RuntimeError("ODE function called with unbatched species but batch conditions > 1.")


        # 1. Calculate symbolic rates
        symbolic_rates = self.symbolic_kinetics(species, encoded_conditions_batch)

        # 2. Calculate neural corrections
        neural_corrections = self.neural_augmentation(species, encoded_conditions_batch)

        # Regularization for neural corrections (can be added to main loss instead)
        self.latest_neural_reg_loss = LAMBDA_RATE_REG * torch.mean(neural_corrections**2)

        # 3. Combine rates
        # Option A: Neural net predicts correction factor (multiplicative)
        # total_rates = symbolic_rates * (1 + neural_corrections) # Needs careful scaling/activation
        # Option B: Neural net predicts additive correction
        total_rates = symbolic_rates + neural_corrections # (batch_size, n_reactions)

        # Ensure total rates are physically plausible (e.g., non-negative if reactions are irreversible)
        total_rates = torch.relu(total_rates) # Assuming forward reactions only

        # 4. Calculate dC/dt using stoichiometry
        # dC/dt = S * R
        # S: (n_species, n_reactions), R: (batch_size, n_reactions)
        # Need S.T: (n_reactions, n_species)
        # dC/dt: (batch_size, n_reactions) @ (n_reactions, n_species) -> (batch_size, n_species)
        dCdt = total_rates @ self.stoichiometry.T # Use transpose of defined S

        # Return derivatives, remove batch dim if input was single sample
        return dCdt.squeeze(0) if not is_batch else dCdt


class Transesterformer(nn.Module):
    """
    The main model class orchestrating the encoder and the Neural ODE.
    """
    def __init__(self, n_conditions=len(CONDITION_COLS),
                 encoder_hidden_dim=ENCODER_HIDDEN_DIM,
                 encoder_output_dim=ENCODER_OUTPUT_DIM,
                 encoder_n_layers=ENCODER_N_LAYERS,
                 node_hidden_dim=NODE_HIDDEN_DIM,
                 node_n_layers=NODE_N_LAYERS,
                 node_activation=NODE_ACTIVATION,
                 ode_solver=None, # Pass solver params from train script
                 ode_options=None):
        super().__init__()
        self.encoder = ConditionEncoder(n_conditions, encoder_hidden_dim, encoder_output_dim, encoder_n_layers)
        self.ode_func = TransesterformerODE(encoder_output_dim, node_hidden_dim, node_n_layers, node_activation)
        self.ode_solver = ode_solver if ode_solver else 'dopri5' # Default solver
        self.ode_options = ode_options if ode_options else {} # Default options

        # Placeholder for regularization loss from ODE func
        self.neural_reg_loss = torch.tensor(0.0, device=DEVICE)


    def forward(self, initial_conditions, times, conditions):
        """
        Performs forward pass: encodes conditions and solves the ODE.
        Args:
            initial_conditions (torch.Tensor): Initial species concentrations C(t=0)
                                               Shape (batch_size, n_species)
            times (torch.Tensor): Time points to evaluate the solution at.
                                  Shape (n_times,) - MUST be sorted.
            conditions (torch.Tensor): Condition vectors for each experiment.
                                       Shape (batch_size, n_conditions)
        Returns:
            torch.Tensor: Predicted species concentrations over time.
                          Shape (batch_size, n_times, n_species)
        """
        # 1. Encode conditions
        encoded_conditions = self.encoder(conditions) # (batch_size, encoded_dim)

        # 2. Set conditions in ODE function for this batch
        self.ode_func.set_conditions(encoded_conditions)

        # 3. Solve the ODE system
        # odeint expects initial conditions (batch, dim), times (times,)
        # It returns (times, batch, dim) -> permute to (batch, times, dim)
        pred_species_over_time = odeint(
            self.ode_func,
            initial_conditions,
            times,
            method=self.ode_solver,
            options=self.ode_options,
            # rtol=self.ode_options.get('rtol', 1e-4), # Pass tolerances via options
            # atol=self.ode_options.get('atol', 1e-4)
        )

        # Retrieve regularization loss calculated during ODE forward calls
        # Note: This might only capture the loss from the last step if not careful.
        # A better way might be to accumulate it within the ODE func or recalculate.
        # For simplicity, we use the last stored value.
        self.neural_reg_loss = self.ode_func.latest_neural_reg_loss if hasattr(self.ode_func, 'latest_neural_reg_loss') else torch.tensor(0.0, device=DEVICE)


        # Permute output to (batch_size, n_times, n_species)
        pred_species_over_time = pred_species_over_time.permute(1, 0, 2)

        return pred_species_over_time


if __name__ == '__main__':
    # Example usage: Instantiate model and run a dummy forward pass
    print(f"Using device: {DEVICE}")
    model = Transesterformer(
        n_conditions=len(CONDITION_COLS),
        ode_options={'rtol': 1e-3, 'atol': 1e-3} # Example options
    ).to(DEVICE)

    print("\nModel Architecture:")
    print(model)

    # Dummy data for one batch of 2 experiments
    batch_size = 2
    n_times = 10
    dummy_initial_c = torch.rand(batch_size, N_SPECIES, device=DEVICE) * 2 # Random initial concentrations
    dummy_times = torch.linspace(0, 5, n_times, device=DEVICE) # 0 to 5 hours
    dummy_conditions = torch.rand(batch_size, len(CONDITION_COLS), device=DEVICE) # Random conditions

    print("\nRunning dummy forward pass...")
    try:
        with torch.no_grad(): # No need to track gradients for this test
            predictions = model(dummy_initial_c, dummy_times, dummy_conditions)
        print(f"Output prediction shape: {predictions.shape}") # Expected: (batch_size, n_times, n_species)
        assert predictions.shape == (batch_size, n_times, N_SPECIES)
        print("Dummy forward pass successful.")
        print(f"Neural regularization loss term: {model.neural_reg_loss.item()}")

    except Exception as e:
        print(f"Error during dummy forward pass: {e}")
        import traceback
        traceback.print_exc()

