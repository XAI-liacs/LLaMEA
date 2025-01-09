import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.pop_size = 30
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w = 0.7   # inertia weight
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_scores = np.full(self.pop_size, float('inf'))
        self.global_best = None
        self.global_best_score = float('inf')

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Evaluate current population
            for i in range(self.pop_size):
                score = func(self.population[i])
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.population[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.population[i]
            
            # PSO update
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            cognitive_component = self.c1 * r1 * (self.personal_best - self.population)
            social_component = self.c2 * r2 * (self.global_best - self.population)
            self.velocities = self.w * self.velocities + cognitive_component + social_component
            self.population += self.velocities
            self.population = np.clip(self.population, self.bounds[0], self.bounds[1])

            # DE mutation and crossover
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                donor_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                donor_vector = np.clip(donor_vector, self.bounds[0], self.bounds[1])
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, 
                                        donor_vector, 
                                        self.population[i])
                
                trial_score = func(trial_vector)
                evaluations += 1
                
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best[i] = trial_vector
                    self.personal_best_scores[i] = trial_score
                    self.population[i] = trial_vector  # Accept trial vector
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best = trial_vector

        return self.global_best