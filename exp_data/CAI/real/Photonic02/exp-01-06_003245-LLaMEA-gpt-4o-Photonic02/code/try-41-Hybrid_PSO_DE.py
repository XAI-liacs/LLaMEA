import numpy as np

class Hybrid_PSO_DE:
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
        
        while eval_count < self.budget:
            for population in self.populations:
                for i in range(self.population_size):
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
                adaptive_vel_range = (self.max_vel - self.min_vel) * (1 - eval_count / self.budget)
                social_coeff_adjusted = self.social_coeff * (1 - eval_count / self.budget)
                
                # Differential Evolution mutation and crossover
                a, b, c = population['positions'][np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), 0, 1)
                crossover = np.random.rand(self.dim) < 0.9
                trial_position = np.where(crossover, mutant, population['positions'][i])
                
                trial_value = func(bounds.lb + trial_position * (bounds.ub - bounds.lb))
                eval_count += 1
                if trial_value < population['personal_best_values'][i]:
                    population['positions'][i] = trial_position
                    population['personal_best_positions'][i] = trial_position
                    population['personal_best_values'][i] = trial_value
                
                population['velocities'] = inertia_adjustment * population['velocities'] + \
                                           self.cognitive_coeff * r1 * (population['personal_best_positions'] - population['positions']) + \
                                           social_coeff_adjusted * r2 * (self.global_best_position - population['positions'])
                
                population['velocities'] = np.clip(population['velocities'], -adaptive_vel_range, adaptive_vel_range)
                population['positions'] += population['velocities']
                population['positions'] = np.clip(population['positions'], 0.0, 1.0)
                
        return bounds.lb + self.global_best_position * (bounds.ub - bounds.lb)