import numpy as np

class QuantumBeesIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 2 * dim)
        self.f = 0.7      # DE Mutation factor
        self.cr = 0.9     # Crossover rate
        self.waggle_rate = 0.1  # Probability of waggle dance communication
        self.beta = 0.05  # Quantum-inspired learning rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_position = population[best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # DE Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.f * (x2 - x3), lb, ub)
                
                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])

                # Quantum-inspired exploration
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=1)
                    trial += q * (best_position - trial)

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Waggle dance communication
                if np.random.rand() < self.waggle_rate:
                    for j in range(self.dim):
                        neighbor = np.random.choice(self.population_size)
                        if fitness[neighbor] < fitness[i]:
                            population[i][j] = np.random.normal(loc=population[neighbor][j], scale=abs(population[i][j] - population[neighbor][j]))

            # Update the best solution found
            best_index = np.argmin(fitness)
            if fitness[best_index] < func(best_position):
                best_position = population[best_index].copy()

        return best_position, fitness[best_index]