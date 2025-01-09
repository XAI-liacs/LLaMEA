import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.pop)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
    
    def __call__(self, func):
        evals = 0
        F = 0.5  # DE scaling factor
        CR = 0.9 # DE crossover probability
        inertia_weight = 0.7
        cognitive_coefficient = 1.5
        social_coefficient = 1.5

        while evals < self.budget:
            # Evaluate population
            for i in range(self.population_size):
                if evals < self.budget:
                    score = func(self.pop[i])
                    evals += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best_positions[i] = self.pop[i]
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.pop[i]

            # PSO update
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = cognitive_coefficient * r1 * (self.personal_best_positions[i] - self.pop[i])
                social_velocity = social_coefficient * r2 * (self.global_best_position - self.pop[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] 
                                      + cognitive_velocity + social_velocity)
                self.pop[i] += self.velocities[i]
                self.pop[i] = np.clip(self.pop[i], self.lower_bound, self.upper_bound)

            # DE mutation and crossover
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.pop[indices]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(self.pop[i])
                crossover_mask = np.random.rand(self.dim) < CR
                trial[crossover_mask] = mutant[crossover_mask]
                trial_score = func(trial)
                evals += 1
                if trial_score < func(self.pop[i]):
                    self.pop[i] = trial

        return self.global_best_position, self.global_best_score