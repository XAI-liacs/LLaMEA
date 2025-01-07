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
        subgroup_count = max(2, self.population_size // 5)  # New: subdivide population

        while eval_count < self.budget:
            for population in self.populations:
                # New: Split population into subgroups for diverse exploration
                np.random.shuffle(population['positions'])
                subgroups = np.array_split(population['positions'], subgroup_count)
                
                for subgroup in subgroups:
                    for i in range(subgroup.shape[0]):
                        scaled_position = bounds.lb + subgroup[i] * (bounds.ub - bounds.lb)
                        current_value = func(scaled_position)
                        eval_count += 1

                        if current_value < population['personal_best_values'][i]:
                            population['personal_best_positions'][i] = subgroup[i]
                            population['personal_best_values'][i] = current_value

                        if current_value < self.global_best_value:
                            self.global_best_position = subgroup[i]
                            self.global_best_value = current_value

                    r1, r2 = np.random.rand(2)
                    inertia_adjustment = self.inertia_weight * (0.9 - (0.8 * eval_count / self.budget))
                    adaptive_vel_range = (self.max_vel - self.min_vel) * (1 - eval_count / self.budget)
                    # New: Introduce diversity measure to adjust social coefficient
                    diversity_factor = np.std(subgroup, axis=0).mean()
                    social_coeff_adjusted = self.social_coeff * (1 - eval_count / self.budget) * diversity_factor
                    
                    population['velocities'] = inertia_adjustment * population['velocities'] + \
                                               self.cognitive_coeff * r1 * (population['personal_best_positions'] - population['positions']) + \
                                               social_coeff_adjusted * r2 * (self.global_best_position - population['positions'])
                    
                    population['velocities'] = np.clip(population['velocities'], -adaptive_vel_range, adaptive_vel_range)
                    population['positions'] += population['velocities']
                    population['positions'] = np.clip(population['positions'], 0.0, 1.0)
                
        return bounds.lb + self.global_best_position * (bounds.ub - bounds.lb)