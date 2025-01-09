import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_positions = np.copy(self.population)
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.f = 0.5  # DE mutation factor
        self.cr = 0.9  # Crossover rate
        self.c1 = 2.05  # PSO cognitive coefficient
        self.c2 = 2.05  # PSO social coefficient
        self.w = 0.9  # PSO inertia weight

    def __call__(self, func):
        evals = 0
        
        while evals < self.budget:
            new_population = np.copy(self.population)
            
            # Differential Evolution step
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(self.population[a] + self.f * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
                
                for j in range(self.dim):
                    if np.random.rand() > self.cr:
                        mutant[j] = self.population[i, j]
                
                # Evaluate mutant
                score = func(mutant)
                evals += 1
                
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = mutant
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = mutant
                
                if evals >= self.budget:
                    break
                
                new_population[i] = mutant if score < func(self.population[i]) else self.population[i]
            
            self.population = new_population
            
            # Particle Swarm Optimization step
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.velocity = self.w * self.velocity + self.c1 * r1 * (self.best_positions - self.population) + self.c2 * r2 * (self.global_best_position - self.population)
            self.population = np.clip(self.population + self.velocity, self.lower_bound, self.upper_bound)
            
            for i in range(self.population_size):
                score = func(self.population[i])
                evals += 1
                
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.population[i]
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]
                
                if evals >= self.budget:
                    break

        return self.global_best_position, self.global_best_score