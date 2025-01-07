import numpy as np

class RefinedQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cross_prob = 0.9
        self.diff_weight = 0.8
        self.sub_population_count = 3  # Introduce multiple sub-populations
        self.adaptive_beta = True  # Flag for adaptive beta

    def initialize_population(self, lb, ub):
        return [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.sub_population_count)]

    def quantum_superposition(self, populations, lb, ub, func):
        quantum_populations = []
        for population in populations:
            if self.adaptive_beta:
                beta = 0.1 / np.sqrt(self.dim) * np.random.rand()  # Adaptive beta
            else:
                beta = 0.18 / np.sqrt(self.dim)
            best_solution = population[np.argmin([func(ind) for ind in population])]
            quantum_population = population + beta * (best_solution - population) * np.random.normal(0, 1, (self.population_size, self.dim))
            np.clip(quantum_population, lb, ub, out=quantum_population)
            quantum_populations.append(quantum_population)
        return quantum_populations

    def differential_evolution(self, populations, lb, ub, func):
        new_populations = []
        for population in populations:
            new_population = np.copy(population)
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                d = np.linalg.norm(b - c) / np.linalg.norm(lb - ub)
                self.diff_weight = 0.3 + 0.6 * np.random.rand() * d
                mutant = np.clip(a + self.diff_weight * (b - c), lb, ub)
                diversity = np.std(population, axis=0).mean()
                self.cross_prob = 0.5 + 0.4 * diversity
                cross_points = np.random.rand(self.dim) < (self.cross_prob * (0.5 + 0.5 * np.random.rand()))
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                if func(trial) < func(population[i]):
                    new_population[i] = trial
            new_populations.append(new_population)
        return new_populations

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        populations = self.initialize_population(lb, ub)
        evaluations = 0
        
        while evaluations < self.budget:
            quantum_populations = self.quantum_superposition(populations, lb, ub, func)
            populations = self.differential_evolution(quantum_populations, lb, ub, func)
            evaluations += self.population_size * self.sub_population_count

        best_individuals = [pop[np.argmin([func(ind) for ind in pop])] for pop in populations]
        best_idx = np.argmin([func(ind) for ind in best_individuals])
        return best_individuals[best_idx]