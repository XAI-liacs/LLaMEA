import numpy as np

class AMP_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.max_vel = 0.2
        self.min_vel = -0.2
        # Changed line: Introduce dynamic population count based on budget
        self.populations = [self._initialize_population(int(self.population_size * (1 + 0.5 * i / (self.budget // self.population_size)))) for i in range(3)]
        self.global_best_position = None
        self.global_best_value = float('inf')
        
    def _initialize_population(self, size):
        return {
            'positions': np.random.uniform(size=(size, self.dim)),
            'velocities': np.random.uniform(low=self.min_vel, high=self.max_vel, size=(size, self.dim)),
            'personal_best_positions': np.zeros((size, self.dim)),
            'personal_best_values': np.full(size, float('inf'))
        }
        
    def __call__(self, func):
        bounds = func.bounds
        eval_count = 0
        # Changed line: Add dynamic inertia weight calculation
        inertia_dynamic_start = 0.9
        inertia_dynamic_end = 0.4
        
        while eval_count < self.budget:
            for population in self.populations:
                for i in range(len(population['positions'])):  # Changed line: Use dynamic population size
                    scaled_position = bounds.lb + population['positions'][i] * (bounds.ub - bounds.lb)
                    current_value = func(scaled_position)
                    eval_count += 1

                    if current_value < population['personal_best_values'][i]:
                        population['personal_best_positions'][i] = population['positions'][i]
                        population['personal_best_values'][i] = current_value

                    if current_value < self.global_best_value:
                        self.global_best_position = population['positions'][i]
                        self.global_best_value = current_value

                r1, r2 = np.random.rand(2)
                # Changed line: Dynamic inertia weight for better exploration-exploitation balance
                inertia_adjustment = inertia_dynamic_start - (inertia_dynamic_start - inertia_dynamic_end) * (eval_count / self.budget)
                adaptive_vel_range = (self.max_vel - self.min_vel) * (1 - eval_count / self.budget)  # Adaptive velocity limit
                social_coeff_adjusted = self.social_coeff * (1 - eval_count / self.budget)
                
                population['velocities'] = inertia_adjustment * population['velocities'] + \
                                           self.cognitive_coeff * r1 * (population['personal_best_positions'] - population['positions']) + \
                                           social_coeff_adjusted * r2 * (self.global_best_position - population['positions'])
                
                population['velocities'] = np.clip(population['velocities'], -adaptive_vel_range, adaptive_vel_range)
                population['positions'] += population['velocities']
                population['positions'] = np.clip(population['positions'], 0.0, 1.0)
                
        return bounds.lb + self.global_best_position * (bounds.ub - bounds.lb)