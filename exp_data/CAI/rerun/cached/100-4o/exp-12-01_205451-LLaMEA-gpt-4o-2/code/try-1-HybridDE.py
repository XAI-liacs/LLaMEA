import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = min(20 + dim, 50)  # Changed to adapt based on dimension
        self.f = 0.7  # differential weight
        self.cr = 0.9  # crossover probability
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                
                # Mutation: select three distinct individuals
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]

                # Generate mutant vector
                mutant = np.clip(x1 + self.f * (x2 - x3), self.bounds[0], self.bounds[1])

                # Crossover: create trial vector
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection: greedy selection between trial and target vector
                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness

            self.population = new_population

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]