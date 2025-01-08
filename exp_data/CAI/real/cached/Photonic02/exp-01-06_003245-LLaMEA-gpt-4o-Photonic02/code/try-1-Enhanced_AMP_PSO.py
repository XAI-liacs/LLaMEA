import numpy as np

class Enhanced_AMP_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.inertia_weight = 0.9  # Adjusted initial inertia weight
        self.cognitive_coeff = 1.7  # Adjusted cognitive coefficient
        self.social_coeff = 1.7  # Adjusted social coefficient
        self.max_vel = 0.3  # Adjusted max velocity
        self.min_vel = -0.3  # Adjusted min velocity
        self.populations = [self._initialize_population() for _ in range(3)]
        self.global_best_position = None
        self.global_best_value = float('inf')
        
    def _initialize_population(self):
        return {
            'positions': np.random.uniform(size=(self.population_size, self.dim)),
            'velocities': np.random.uniform(low=self.min_vel, high=self.max_vel, size=(self.population_size, self.dim)),
            'personal_best_positions': np.zeros((self.population_size, self.dim)),
            'personal_best_values': np.full(self.population_size, float('inf'))
        }
        
    def __call__(self, func):
        bounds = func.bounds
        eval_count = 0
        
        while eval_count < self.budget:
            for population in self.populations:
                for i in range(self.population_size):
                    # Scale position within bounds
                    scaled_position = bounds.lb + population['positions'][i] * (bounds.ub - bounds.lb)
                    current_value = func(scaled_position)
                    eval_count += 1

                    # Update personal best
                    if current_value < population['personal_best_values'][i]:
                        population['personal_best_positions'][i] = population['positions'][i]
                        population['personal_best_values'][i] = current_value

                    # Update global best
                    if current_value < self.global_best_value:
                        self.global_best_position = population['positions'][i]
                        self.global_best_value = current_value

                inertia_adjustment = self.inertia_weight - (0.5 * eval_count / self.budget)  # Dynamic inertia adjustment
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(2)
                    selected_idx = np.random.choice(range(self.population_size))  # Competition-based selection
                    population['velocities'][i] = inertia_adjustment * population['velocities'][i] + \
                                                  self.cognitive_coeff * r1 * (population['personal_best_positions'][i] - population['positions'][i]) + \
                                                  self.social_coeff * r2 * (population['personal_best_positions'][selected_idx] - population['positions'][i])  # Use selected competitor
                    
                    population['velocities'][i] = np.clip(population['velocities'][i], self.min_vel, self.max_vel)
                    population['positions'][i] += population['velocities'][i]
                    population['positions'][i] = np.clip(population['positions'][i], 0.0, 1.0)
                
        # Return best solution found
        return bounds.lb + self.global_best_position * (bounds.ub - bounds.lb)