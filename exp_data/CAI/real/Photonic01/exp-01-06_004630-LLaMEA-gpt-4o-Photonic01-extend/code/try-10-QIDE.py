import numpy as np

class QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cross_prob = 0.9
        self.diff_weight = 0.8

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def quantum_superposition(self, population, lb, ub, func):
        beta = 0.1 / np.sqrt(self.dim)  # Changed from 0.05 to make beta adaptive to dimensionality
        best_solution = population[np.argmin([func(ind) for ind in population])]
        quantum_population = population + beta * (best_solution - population) * np.random.normal(0, 1, (self.population_size, self.dim))
        np.clip(quantum_population, lb, ub, out=quantum_population)
        return quantum_population

    def differential_evolution(self, population, lb, ub, func):
        new_population = np.copy(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            self.diff_weight = 0.5 + 0.3 * np.random.rand()  
            mutant = np.clip(a + self.diff_weight * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if func(trial) < func(population[i]):
                new_population[i] = trial
        return new_population

    def local_search(self, individual, lb, ub, func):
        step_size = 0.05 * (ub - lb)
        candidate = np.clip(individual + step_size * np.random.uniform(-1, 1, self.dim), lb, ub)
        if func(candidate) < func(individual):
            return candidate
        return individual

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            quantum_population = self.quantum_superposition(population, lb, ub, func)
            population = self.differential_evolution(quantum_population, lb, ub, func)
            population = np.array([self.local_search(ind, lb, ub, func) for ind in population])
            evaluations += self.population_size

        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]

# Example of use:
# optimizer = QIDE(budget=1000, dim=10)
# best_solution = optimizer(func)