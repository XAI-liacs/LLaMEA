import numpy as np

class AMP_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.max_vel = 0.2
        self.min_vel = -0.2
        self.populations = [self._initialize_population() for _ in range(3)]
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.f = 0.5  # Differential weight
        self.cr = 0.7  # Crossover probability
        
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
                    pos = population['positions'][i]
                    scaled_position = bounds.lb + pos * (bounds.ub - bounds.lb)
                    current_value = func(scaled_position)
                    eval_count += 1

                    if current_value < population['personal_best_values'][i]:
                        population['personal_best_positions'][i] = pos
                        population['personal_best_values'][i] = current_value

                    if current_value < self.global_best_value:
                        self.global_best_position = pos
                        self.global_best_value = current_value

                r1, r2 = np.random.rand(2)
                inertia_adjustment = 0.6 + 0.3 * np.cos(np.pi * eval_count / self.budget)
                adaptive_vel_range = (self.max_vel - self.min_vel) * (1 - eval_count / self.budget)
                
                # Differential Evolution Mutation
                for j in range(self.population_size):
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    x1, x2, x3 = population['positions'][idxs]
                    mutant = x1 + self.f * (x2 - x3)
                    trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population['positions'][j])
                    population['positions'][j] = np.clip(trial, 0.0, 1.0)
                
                population['velocities'] = inertia_adjustment * population['velocities'] + \
                                           self.cognitive_coeff * r1 * (population['personal_best_positions'] - population['positions']) + \
                                           self.social_coeff * r2 * (self.global_best_position - population['positions'])
                
                population['velocities'] = np.clip(population['velocities'], -adaptive_vel_range, adaptive_vel_range)
                population['positions'] += population['velocities']
                population['positions'] = np.clip(population['positions'], 0.0, 1.0)
                
        return bounds.lb + self.global_best_position * (bounds.ub - bounds.lb)