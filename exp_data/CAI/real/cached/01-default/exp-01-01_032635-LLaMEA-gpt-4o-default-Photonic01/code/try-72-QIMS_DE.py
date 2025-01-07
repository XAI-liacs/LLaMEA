import numpy as np

class QIMS_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.adapt_rate = 0.1
        self.positions = None
        self.best_position = None
        self.best_value = np.inf

    def quantum_initialize(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        beta = np.random.normal(0, 1, (self.population_size, self.dim))
        self.positions += beta * (ub - lb) * 0.1
        self.positions = np.clip(self.positions, lb, ub)

    def differential_mutation(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        return np.clip(mutant, *self.bounds)

    def crossover(self, target, mutant):
        j_random = np.random.randint(self.dim)
        trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, target)
        trial[j_random] = mutant[j_random]
        return trial

    def adapt_mutation_factor(self):
        diversity = np.mean(np.std(self.positions, axis=0))
        self.mutation_factor = 0.5 + 0.5 * (diversity / np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2)))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.bounds = (lb, ub)
        self.quantum_initialize(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant = self.differential_mutation(i)
                trial = self.crossover(self.positions[i], mutant)
                
                trial_value = func(trial)
                evaluations += 1

                current_value = func(self.positions[i])
                evaluations += 1

                if trial_value < current_value:
                    self.positions[i] = trial
                    current_value = trial_value

                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_position = self.positions[i].copy()

            self.adapt_mutation_factor()

        return self.best_position, self.best_value