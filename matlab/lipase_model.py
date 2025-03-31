import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class LipaseKineticModel:
    """Enhanced model for lipase-catalyzed transesterification kinetics prediction
    
    Implements analytical solutions and advanced parameter optimization
    for pseudo-first-order reaction systems"""
    
    def __init__(self, C0=1.0, t_end=10, num_points=100):
        """
        Initialize kinetic model parameters
        :param C0: Initial substrate concentration (mol/L)
        :param t_end: Reaction endpoint (min)
        :param num_points: Number of time points for simulation
        """
        self.C0 = C0
        self.t = np.linspace(0, t_end, num_points)
        self.R = 8.314  # Universal gas constant (J/mol·K)
        
        # Empirical parameters (expandable through parameter estimation)
        self.Ea = 5000    # Activation energy (J/mol)
        self.k_ref = 0.1  # Reference rate constant at standard conditions
        
    def _arrhenius(self, T, Ea_factor=1.0):
        """Temperature dependence with adjustable activation energy scaling"""
        return np.exp(-Ea_factor*self.Ea / (self.R*(T + 273.15)))
    
    def reaction_rate_constant(self, enzyme_loading, T, methanol_ratio):
        """
        Enhanced rate constant calculation incorporating:
        - Nonlinear enzyme loading effects
        - Substrate inhibition effects
        - Temperature-dependent activation energy
        """
        # Substrate inhibition factor (empirical relation)
        inhibition_factor = 1 / (1 + (methanol_ratio/4)**2)
        
        # Temperature compensation factor
        arrhenius_factor = self._arrhenius(T)
        
        return (self.k_ref * enzyme_loading**0.7 * arrhenius_factor 
                * inhibition_factor * np.log1p(methanol_ratio))

    def concentration_profile(self, k):
        """Analytical solution for pseudo-first-order kinetics"""
        return self.C0 * np.exp(-k * self.t)
    
    def conversion_efficiency(self, k, t_end=None):
        """Calculate percentage conversion using analytical solution"""
        t_final = t_end if t_end else self.t[-1]
        return 100 * (1 - np.exp(-k * t_final))
    
    def multi_param_sensitivity(self, params, variations=5):
        """
        Perform multivariate sensitivity analysis across parameters:
        - Enzyme loading (g/L)
        - Temperature (°C)
        - Methanol ratio (mol:mol)
        """
        results = []
        k_values = []
        
        # Generate parameter space using meshgrid
        enzyme_space = np.linspace(*params['enzyme_loading'], variations)
        temp_space = np.linspace(*params['temperature'], variations)
        ratio_space = np.linspace(*params['methanol_ratio'], variations)
        
        # Vectorized calculation of rate constants
        E, T, M = np.meshgrid(enzyme_space, temp_space, ratio_space)
        k_matrix = self.reaction_rate_constant(E, T, M)
        
        # Calculate concentration profiles for all combinations
        conc_profiles = self.concentration_profile(k_matrix.reshape(-1,1))
        
        return {
            'enzyme': E.ravel(),
            'temperature': T.ravel(),
            'ratio': M.ravel(),
            'k_values': k_matrix.ravel(),
            'profiles': conc_profiles
        }
    
    def optimize_conditions(self, param_bounds, target_time=10):
        """
        Optimize process parameters for maximum conversion
        :param param_bounds: Dictionary of parameter bounds:
            {
                'enzyme_loading': (min, max),
                'temperature': (min, max),
                'methanol_ratio': (min, max)
            }
        :param target_time: Target reaction time (min)
        """
        def objective(x):
            enzyme, temp, ratio = x
            k = self.reaction_rate_constant(enzyme, temp, ratio)
            return -self.conversion_efficiency(k, target_time)
        
        initial_guess = [
            np.mean(param_bounds['enzyme_loading']),
            np.mean(param_bounds['temperature']),
            np.mean(param_bounds['methanol_ratio'])
        ]
        
        bounds = [
            param_bounds['enzyme_loading'],
            param_bounds['temperature'],
            param_bounds['methanol_ratio']
        ]
        
        result = minimize(objective, initial_guess, bounds=bounds,
                          method='L-BFGS-B', 
                          options={'maxiter': 100, 'disp': True})
        
        return {
            'optimal_conditions': result.x,
            'conversion': -result.fun,
            'rate_constant': self.reaction_rate_constant(*result.x)
        }

# Example Usage
if __name__ == "__main__":
    # Initialize model with expanded parameter ranges
    model = LipaseKineticModel(C0=2.0, t_end=60, num_points=200)
    
    # Perform multivariate sensitivity analysis
    param_ranges = {
        'enzyme_loading': (0.1, 20.0),    # g/L
        'temperature': (30, 70),         # °C
        'methanol_ratio': (1, 5)         # mol:mol
    }
    
    sensitivity_results = model.multi_param_sensitivity(param_ranges)
    
    # Visualization of results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(sensitivity_results['enzyme'],
                    sensitivity_results['temperature'],
                    sensitivity_results['ratio'],
                    c=sensitivity_results['k_values'],
                    cmap='viridis', s=50)
    
    ax.set_xlabel('Enzyme Loading (g/L)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_zlabel('Methanol Ratio')
    fig.colorbar(sc, label='Rate Constant (1/min)')
    plt.title('Multivariate Parameter Sensitivity Analysis')
    plt.show()
    
    # Optimization for maximum conversion
    optimization_bounds = {
        'enzyme_loading': (0.1, 10.0),    # g/L
        'temperature': (30, 70),         # °C
        'methanol_ratio': (1, 5)         # mol:mol
    }

    
    opt_results = model.optimize_conditions(optimization_bounds)
    
    
    print(f"\nOptimal Conditions:")
    print(f"Enzyme Loading: {opt_results['optimal_conditions'][0]:.2f} g/L")
    print(f"Temperature: {opt_results['optimal_conditions'][1]:.1f} °C")
    print(f"Methanol Ratio: {opt_results['optimal_conditions'][2]:.1f} mol:mol")
    print(f"Predicted Conversion: {opt_results['conversion']:.1f}%")
    print(f"Resulting Rate Constant: {opt_results['rate_constant']:.4f} 1/min")
    
