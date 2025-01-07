import numpy as np

class QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cross_prob = 0.85  # Reduced for better exploitation
        self.diff_weight = 0.7  # Reduced for balanced mutation

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def quantum_superposition(self, population, lb, ub, func):
        beta = 0.18 / np.sqrt(self.dim)
        best_solution = population[np.argmin([func(ind) for ind in population])]
        quantum_population = population + beta * (best_solution - population) * np.random.normal(0, 1, (self.population_size, self.dim))
        np.clip(quantum_population, lb, ub, out=quantum_population)
        return quantum_population

    def differential_evolution(self, population, lb, ub, func):
        new_population = np.copy(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            d = np.linalg.norm(b - c) / np.linalg.norm(lb - ub)
            self.diff_weight = 0.3 + 0.5 * np.random.rand() * d  # Adjusted weight range
            mutant = np.clip(a + self.diff_weight * (b - c), lb, ub)
            diversity = np.std(population, axis=0).mean()
            self.cross_prob = 0.55 + 0.3 * diversity  # Adjusted cross probability
            cross_points = np.random.rand(self.dim) < (self.cross_prob * (0.5 + 0.5 * np.random.rand()))
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if func(trial) < func(population[i]):
                new_population[i] = trial
            # Local search around the best solution
            if np.random.rand() < 0.1:
                local_solution = best_solution + 0.05 * np.random.normal(0, 1, self.dim)
                np.clip(local_solution, lb, ub, out=local_solution)
                if func(local_solution) < func(best_solution):
                    new_population[i] = local_solution
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