import numpy as np

# Default parameter values
DEFAULT_PARAMS = {
    'L1': 1.0,    # Length of link 1 (m)
    'L2': 1.0,    # Length of link 2 (m)
    'm1': 1.0,    # Mass of link 1 (kg)
    'm2': 1.0,    # Mass of link 2 (kg)
    'g': 9.81,    # Gravity (m/s^2)
    'tau1': 0.0,  # Torque at joint 1 (N⋅m)
    'tau2': 0.0,  # Torque at joint 2 (N⋅m)
}

# Default initial conditions
DEFAULT_INITIAL_CONDITIONS = {
    'theta1_0': np.pi/4,    # Initial angle of link 1 (rad)
    'theta2_0': np.pi/4,    # Initial angle of link 2 (rad)
    'omega1_0': 0.0,        # Initial angular velocity of link 1 (rad/s)
    'omega2_0': 0.0         # Initial angular velocity of link 2 (rad/s)
}

# Simulation parameters
DEFAULT_SIM_PARAMS = {
    't_start': 0.0,   # Start time (s)
    't_end': 10.0,    # End time (s)
    'dt': 0.02        # Time step (s)
}

class Parameters:
    # Parameter limits for UI sliders
    PARAM_LIMITS = {
        'L1': (0.1, 2.0),      # Link 1 length limits (m)
        'L2': (0.1, 2.0),      # Link 2 length limits (m)
        'm1': (0.1, 5.0),      # Link 1 mass limits (kg)
        'm2': (0.1, 5.0),      # Link 2 mass limits (kg)
        'g': (0.0, 15.0),      # Gravity limits (m/s^2)
        'tau1': (-10.0, 10.0), # Joint 1 torque limits (N⋅m)
        'tau2': (-10.0, 10.0), # Joint 2 torque limits (N⋅m)
        'theta1_0': (-np.pi, np.pi),   # Initial angle 1 limits (rad)
        'theta2_0': (-np.pi, np.pi),   # Initial angle 2 limits (rad)
        'omega1_0': (-5.0, 5.0),       # Initial angular velocity 1 limits (rad/s)
        'omega2_0': (-5.0, 5.0)        # Initial angular velocity 2 limits (rad/s)
    }
    
    def __init__(self):
        # Initialize with default values
        self._params = DEFAULT_PARAMS.copy()
        self._initial_conditions = DEFAULT_INITIAL_CONDITIONS.copy()
        self._sim_params = DEFAULT_SIM_PARAMS.copy()
        
    def update_param(self, param_name, value):
        """Update a parameter value within its limits."""
        if param_name in DEFAULT_PARAMS:
            if param_name in self.PARAM_LIMITS:
                min_val, max_val = self.PARAM_LIMITS[param_name]
                value = np.clip(value, min_val, max_val)
            self._params[param_name] = value
        elif param_name in DEFAULT_INITIAL_CONDITIONS:
            if param_name in self.PARAM_LIMITS:
                min_val, max_val = self.PARAM_LIMITS[param_name]
                value = np.clip(value, min_val, max_val)
            self._initial_conditions[param_name] = value
        elif param_name in DEFAULT_SIM_PARAMS:
            self._sim_params[param_name] = value
    
    @property
    def params(self):
        """Get current physical parameters."""
        return self._params.copy()
    
    @property
    def initial_conditions(self):
        """Get current initial conditions."""
        return self._initial_conditions.copy()
    
    @property
    def sim_params(self):
        """Get current simulation parameters."""
        return self._sim_params.copy()
    
    def get_initial_state_vector(self):
        """Return initial conditions as a state vector [theta1, theta2, omega1, omega2]."""
        return [
            self._initial_conditions['theta1_0'],
            self._initial_conditions['theta2_0'],
            self._initial_conditions['omega1_0'],
            self._initial_conditions['omega2_0']
        ]