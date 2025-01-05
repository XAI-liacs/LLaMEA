import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize quantum bits
        qbits = np.random.uniform(-1, 1, (self.population_size, self.dim))
        evaluations = 0

        while evaluations < self.budget:
            # Measure population from quantum bits
            pop = np.sign(np.random.uniform(-1, 1, (self.population_size, self.dim))) * qbits
            pop = np.clip(pop, lb, ub)
            fitness = np.array([func(x) for x in pop])
            evaluations += self.population_size

            # Update best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_fitness = fitness[min_idx]
                self.best_solution = pop[min_idx]

            # Adaptive quantum rotation gates
            avg_fitness = np.mean(fitness)
            rotation_angles = 0.1 * np.pi * (fitness - avg_fitness) / (np.max(fitness) - avg_fitness + 1e-9)
            qbits += rotation_angles[:, np.newaxis] * np.sign(pop - self.best_solution)

            # Dynamic population control
            diversity = np.mean(np.std(pop, axis=0))
            if diversity < 0.1:
                qbits = np.random.uniform(-1, 1, (self.population_size, self.dim))

            # Save the history of best solutions
            self.history.append(self.best_solution)
        
        return self.best_solution