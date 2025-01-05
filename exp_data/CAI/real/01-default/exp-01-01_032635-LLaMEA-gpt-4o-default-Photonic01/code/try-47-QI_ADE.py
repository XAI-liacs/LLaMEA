import numpy as np

class QI_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget)
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = np.inf
        self.cr = 0.9  # Crossover rate
        self.f = 0.8  # Differential weight
        self.beta = 0.1  # Quantum influence factor
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_update(self, target, best):
        direction = best - target
        quantum_shift = self.beta * np.random.normal(0, 1, self.dim)
        new_position = target + direction * quantum_shift
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def mutate(self, i):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        while i in indices:
            indices = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = self.population[indices]
        mutant = a + self.f * (b - c)
        lb, ub = self.bounds
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                current_fitness = func(self.population[i])
                evaluations += 1

                if current_fitness < self.fitness[i]:
                    self.fitness[i] = current_fitness

                if current_fitness < self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_individual = self.population[i].copy()

            for i in range(self.pop_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_individual = trial.copy()

                # Quantum-inspired update
                self.population[i] = self.quantum_update(self.population[i], self.best_individual)

        return self.best_individual, self.best_fitness