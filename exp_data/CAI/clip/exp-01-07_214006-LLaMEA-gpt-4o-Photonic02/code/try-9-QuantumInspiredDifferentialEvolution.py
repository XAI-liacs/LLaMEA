import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.current_evaluations = 0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.beta_min = 0.5  # Minimum quantum shift
        self.beta_max = 1.0  # Maximum quantum shift

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += self.population_size

        while self.current_evaluations < self.budget:
            new_population = np.copy(population)
            
            for i in range(self.population_size):
                # Mutation
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[idxs]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, bounds[:,0], bounds[:,1])

                # Quantum-inspired crossover
                beta = self.beta_min + (self.beta_max - self.beta_min) * (self.current_evaluations / self.budget)
                crossover = np.random.rand(self.dim) < self.CR
                quantum_shift = beta * (np.random.rand(self.dim) - 0.5)
                trial = np.where(crossover, mutant + quantum_shift, population[i])
                trial = np.clip(trial, bounds[:,0], bounds[:,1])

                # Selection
                trial_fitness = func(trial)
                self.current_evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                if self.current_evaluations >= self.budget:
                    break
            
            population = new_population

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]