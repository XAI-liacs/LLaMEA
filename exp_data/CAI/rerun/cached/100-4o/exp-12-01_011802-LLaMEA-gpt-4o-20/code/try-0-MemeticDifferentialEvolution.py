import numpy as np

class MemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_value = np.inf

    def __call__(self, func):
        evaluations = 0
        fitness = np.apply_along_axis(func, 1, self.population)
        self.best_solution = self.population[np.argmin(fitness)]
        self.best_value = np.min(fitness)
        evaluations += self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, self.population[i])
                
                # Local Search (Hill Climbing)
                trial_improved = trial.copy()
                for j in range(self.dim):
                    perturb = np.zeros(self.dim)
                    perturb[j] = (np.random.rand() - 0.5) / 10.0
                    candidate = np.clip(trial + perturb, self.lower_bound, self.upper_bound)
                    candidate_value = func(candidate)
                    evaluations += 1
                    if candidate_value < func(trial_improved):
                        trial_improved = candidate
                        if candidate_value < self.best_value:
                            self.best_solution = candidate
                            self.best_value = candidate_value
                    if evaluations >= self.budget:
                        break

                trial_value = func(trial_improved)
                evaluations += 1
                if trial_value < fitness[i]:
                    self.population[i] = trial_improved
                    fitness[i] = trial_value
                    if trial_value < self.best_value:
                        self.best_solution = trial_improved
                        self.best_value = trial_value

        return self.best_solution, self.best_value