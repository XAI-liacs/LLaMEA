import numpy as np

class Adaptive_Differential_Evolution_Quantum_Levy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.F = 0.7
        self.CR = 0.9
        self.adaptive_scale = 0.5
        self.q_factor = 0.1
        self.levy_alpha = 1.5
        self.levy_beta = 0.007 

    def levy_flight(self, u):
        num = np.random.normal(0, self.levy_beta, self.dim)
        den = np.power(np.abs(np.random.normal(0, 1, self.dim)), 1 / self.levy_alpha)
        step = num / den
        return u + step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_pos = population[best_idx]
        best_val = fitness[best_idx]

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial = self.levy_flight(trial)
                trial = np.clip(trial, lb, ub)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_val:
                        best_pos = trial
                        best_val = trial_fitness

                if evaluations >= self.budget:
                    break

            self.F *= self.adaptive_scale  # Dynamically adjust differential weight

        return best_pos, best_val