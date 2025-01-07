import numpy as np

class Enhanced_AMP_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.initial_population_size = self.population_size
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.max_vel = 0.2
        self.min_vel = -0.2
        self.populations = [self._initialize_population() for _ in range(3)]
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.tournament_size = max(2, int(0.1 * self.population_size))

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
        no_improvement_count = 0
        
        while eval_count < self.budget:
            for population in self.populations:
                improved = False
                for i in range(self.population_size):
                    scaled_position = bounds.lb + population['positions'][i] * (bounds.ub - bounds.lb)
                    current_value = func(scaled_position)
                    eval_count += 1

                    if current_value < population['personal_best_values'][i]:
                        population['personal_best_positions'][i] = population['positions'][i]
                        population['personal_best_values'][i] = current_value
                        improved = True

                    if current_value < self.global_best_value:
                        self.global_best_position = population['positions'][i]
                        self.global_best_value = current_value
                
                if no_improvement_count >= 0.2 * self.budget:
                    reset_indices = np.random.choice(self.population_size, size=int(0.1 * self.population_size), replace=False)
                    population['positions'][reset_indices] = np.random.uniform(size=(len(reset_indices), self.dim))
                    no_improvement_count = 0

                if not improved:
                    no_improvement_count += 1
                    if no_improvement_count >= 0.1 * self.budget:
                        self.population_size = max(10, self.population_size // 2)
                        population['positions'] = population['positions'][:self.population_size]
                        population['velocities'] = population['velocities'][:self.population_size]
                        population['personal_best_positions'] = population['personal_best_positions'][:self.population_size]
                        population['personal_best_values'] = population['personal_best_values'][:self.population_size]
                else:
                    no_improvement_count = 0
                    if self.population_size < self.initial_population_size:
                        self.population_size = min(self.initial_population_size, self.population_size + 1)

                r1, r2 = np.random.rand(2)
                inertia_adjustment = self.inertia_weight * (0.9 - (0.8 * eval_count / self.budget))
                adaptive_vel_range = (self.max_vel - self.min_vel) * (1 - eval_count / self.budget)
                social_coeff_adjusted = self.social_coeff * (1 - eval_count / self.budget)

                cognitive_coeff_adjusted = self.cognitive_coeff * (0.5 + 0.5 * (1 - eval_count / self.budget))
                
                tournament_indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
                tournament_best_index = np.argmin(population['personal_best_values'][tournament_indices])
                tournament_best_position = population['personal_best_positions'][tournament_indices][tournament_best_index]
                
                population['velocities'] = inertia_adjustment * population['velocities'] + \
                                           cognitive_coeff_adjusted * r1 * (population['personal_best_positions'] - population['positions']) + \
                                           social_coeff_adjusted * r2 * (tournament_best_position - population['positions'])
                
                population['velocities'] = np.clip(population['velocities'], -adaptive_vel_range, adaptive_vel_range)
                population['positions'] += population['velocities']
                population['positions'] = np.clip(population['positions'], 0.0, 1.0)
                
        return bounds.lb + self.global_best_position * (bounds.ub - bounds.lb)