import numpy as np

class DEQFL:
    def __init__(self, budget, dim, population_size=20, mutation_factor=0.8, crossover_prob=0.9, levy_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.levy_scale = levy_scale
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        best_idx = np.argmin(fitness)
        best_value = fitness[best_idx]
        best_solution = population[best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                trial = self.mutate(i, population, lb, ub)
                trial = self.crossover(population[i], trial)
                trial_value = func(trial)
                self.evaluations += 1

                if trial_value < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_value
                    if trial_value < best_value:
                        best_value = trial_value
                        best_solution = trial

            population = self.quantum_levy_flight(population, lb, ub, best_solution)

        return best_solution

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, target_idx, population, lb, ub):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return np.clip(mutant, lb, ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        if not crossover_mask.any():
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, mutant, target)

    def quantum_levy_flight(self, population, lb, ub, best_solution):
        alpha = 1.5  # Levy exponent
        for i in range(self.population_size):
            step = self.levy_scale * self.levy_flight(alpha, self.dim) * (population[i] - best_solution)
            new_position = np.clip(population[i] + step, lb, ub)
            population[i] = new_position
        return population

    def levy_flight(self, alpha, dim):
        sigma_u = (np.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
                   (np.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)
        return u / np.abs(v)**(1 / alpha)