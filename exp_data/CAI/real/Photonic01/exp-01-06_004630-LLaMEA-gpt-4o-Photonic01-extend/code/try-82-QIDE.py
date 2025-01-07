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
        beta = 0.18 / np.sqrt(self.dim)  # Adjusted beta for improved balance
        best_solution = population[np.argmin([func(ind) for ind in population])]
        quantum_population = population + beta * (best_solution - population) * np.random.normal(0, 1, (self.population_size, self.dim))
        np.clip(quantum_population, lb, ub, out=quantum_population)
        return quantum_population

    def differential_evolution(self, population, lb, ub, func):
        new_population = np.copy(population)
        fitness_values = np.array([func(ind) for ind in population])  # Calculate fitness once
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            d = np.linalg.norm(b - c) / np.linalg.norm(lb - ub)
            self.diff_weight = 0.3 + 0.6 * np.random.rand() * d  # Adjust weight range for dynamic scaling
            mutant = np.clip(a + self.diff_weight * (b - c), lb, ub)
            diversity = np.std(fitness_values)  # Use fitness variance for crossover probability
            self.cross_prob = 0.5 + 0.4 * (1 - diversity)  # Adaptive crossover based on fitness variance
            cross_points = np.random.rand(self.dim) < (self.cross_prob * (0.5 + 0.5 * np.random.rand()))
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if func(trial) < fitness_values[i]:
                new_population[i] = trial
        return new_population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0
        
        while evaluations < self.budget:
            quantum_population = self.quantum_superposition(population, lb, ub, func)
            population = self.differential_evolution(quantum_population, lb, ub, func)
            evaluations += self.population_size

        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]