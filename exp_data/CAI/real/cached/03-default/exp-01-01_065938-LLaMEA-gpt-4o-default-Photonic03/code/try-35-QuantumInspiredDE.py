import numpy as np

class QuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Initial scaling factor
        self.CR = 0.9  # Initial crossover rate
        self.alpha = np.pi / 4  # Initial quantum rotation angle

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]

                # Quantum-inspired rotation
                theta = self.alpha * np.random.uniform(-1, 1, self.dim)
                quantum_mutant = x0 + self.F * (x1 - x2) * np.cos(theta) + np.sin(theta) * (best_global - x0)

                # Quantum crossover
                r = np.random.rand(self.dim)
                quantum_trial = np.where(r < self.CR, quantum_mutant, pop[i])
                quantum_trial = np.clip(quantum_trial, lb, ub)
                quantum_fitness = func(quantum_trial)
                evaluations += 1

                if quantum_fitness < fitness[i]:
                    next_pop[i] = quantum_trial
                    fitness[i] = quantum_fitness
                    if quantum_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = quantum_trial
                else:
                    next_pop[i] = pop[i]

            # Adapt F and CR
            self.F = np.clip(self.F + 0.1 * (np.random.rand() - 0.5), 0.4, 0.9)
            self.CR = np.clip(self.CR + 0.1 * (np.random.rand() - 0.5), 0.7, 1.0)
            # Adapt quantum rotation angle
            self.alpha = np.clip(self.alpha + 0.05 * (np.random.rand() - 0.5), np.pi/6, np.pi/3)

            pop = next_pop

        return best_global