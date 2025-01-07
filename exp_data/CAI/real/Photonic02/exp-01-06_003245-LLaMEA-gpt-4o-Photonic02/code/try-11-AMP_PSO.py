import numpy as np

class AMP_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = max(10, 5 * dim)
        self.min_population_size = 5  # New parameter for minimum population size
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.max_vel = 0.2
        self.min_vel = -0.2
        self.populations = [self._initialize_population(self.initial_population_size) for _ in range(3)]
        self.global_best_position = None
        self.global_best_value = float('inf')
        
    def _initialize_population(self, size):  # Adjusted to take dynamic size
        return {
            'positions': np.random.uniform(size=(size, self.dim)),
            'velocities': np.random.uniform(low=self.min_vel, high=self.max_vel, size=(size, self.dim)),
            'personal_best_positions': np.zeros((size, self.dim)),
            'personal_best_values': np.full(size, float('inf'))
        }
        
    def __call__(self, func):
        bounds = func.bounds
        eval_count = 0
        population_size = self.initial_population_size
        
        while eval_count < self.budget:
            for population in self.populations:
                for i in range(population_size):
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
                inertia_adjustment = 0.9 - (0.8 * eval_count / self.budget)
                adaptive_vel_range = self.max_vel * (1 - eval_count / self.budget)  # Simplified adaptive velocity limit
                population['velocities'] = inertia_adjustment * population['velocities'] + \
                                           self.cognitive_coeff * r1 * (population['personal_best_positions'] - population['positions']) + \
                                           self.social_coeff * r2 * (self.global_best_position - population['positions'])

                population['velocities'] = np.clip(population['velocities'], -adaptive_vel_range, adaptive_vel_range)
                population['positions'] += population['velocities']
                population['positions'] = np.clip(population['positions'], 0.0, 1.0)
                
            if eval_count % (self.budget // 10) == 0:  # Dynamic population resizing
                population_size = max(self.min_population_size, int(population_size * 0.9))
                self.populations = [self._initialize_population(population_size) for _ in range(3)]
                
        return bounds.lb + self.global_best_position * (bounds.ub - bounds.lb)