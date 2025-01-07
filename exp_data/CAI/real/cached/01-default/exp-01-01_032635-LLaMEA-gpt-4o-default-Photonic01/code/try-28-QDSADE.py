import numpy as np

class QDSADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.bounds = None
        self.global_best_position = None
        self.global_best_value = np.inf

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.bounds = (lb, ub)

    def self_adaptive_parameters(self):
        # Self-adaptive strategy for mutation factor and crossover rate
        self.mutation_factor = np.random.uniform(0.4, 0.9)
        self.crossover_rate = np.random.uniform(0.3, 0.9)

    def quantum_mutation(self, target, best, lb, ub):
        # Quantum-inspired mutation using Gaussian perturbation
        beta = np.random.normal(0, 1, self.dim)
        mutant = target + self.mutation_factor * (best - target) + beta * 0.1
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        # Binomial crossover
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate:
                trial[j] = mutant[j]
        return trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0
        
        while evaluations < self.budget:
            self.self_adaptive_parameters()
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                idxs = list(range(i)) + list(range(i+1, self.population_size))
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]

                best_idx = np.argmin([func(ind) for ind in self.population])
                best = self.population[best_idx]

                mutant = self.quantum_mutation(self.population[i], best, lb, ub)
                trial = self.crossover(self.population[i], mutant)

                target_value = func(self.population[i])
                trial_value = func(trial)
                evaluations += 2

                if trial_value < target_value:
                    self.population[i] = trial
                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best_position = trial.copy()

        return self.global_best_position, self.global_best_value