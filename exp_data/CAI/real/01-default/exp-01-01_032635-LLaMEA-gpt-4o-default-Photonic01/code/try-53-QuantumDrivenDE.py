import numpy as np

class QuantumDrivenDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.bounds = None
        self.population = None
        self.best_individual = None
        self.best_value = np.inf

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.bounds = (lb, ub)

    def quantum_update(self, individual, best_individual):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim)
        new_position = individual + beta * (best_individual - individual) + delta * 0.1
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def differential_mutation(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        lb, ub = self.bounds
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        jrand = np.random.randint(self.dim)
        trial = np.zeros(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_prob or j == jrand:
                trial[j] = mutant[j]
            else:
                trial[j] = target[j]
        return trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self.differential_mutation(i)
                trial = self.crossover(target, mutant)

                trial_value = func(trial)
                evaluations += 1

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_individual = trial.copy()

                target_value = func(target)
                if trial_value < target_value:
                    self.population[i] = trial

                # Quantum-inspired update
                if np.random.rand() < 0.2:
                    self.population[i] = self.quantum_update(self.population[i], self.best_individual)

        return self.best_individual, self.best_value