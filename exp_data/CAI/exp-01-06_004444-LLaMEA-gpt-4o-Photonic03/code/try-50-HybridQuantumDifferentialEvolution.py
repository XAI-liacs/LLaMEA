import numpy as np

class HybridQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.memory_size = 5
        self.memory = []

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def quantum_inspired_mutation(self, individual, best_solution, lb, ub):
        dynamic_factor = (self.budget / (self.budget + 5)) * 1.01  # Adjusted line for increased dynamic factor
        amplitude = np.abs(np.random.normal(0, dynamic_factor, self.dim))
        mutated = individual + amplitude * (best_solution - individual)
        return np.clip(mutated, lb, ub)

    def differential_crossover(self, target, donor, cr):
        mask = np.random.rand(self.dim) < cr
        trial = np.where(mask, donor, target)
        return trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        while self.budget > 0:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]

            new_population = []
            new_fitness = []

            fitness_variance = np.var(fitness)  # New line to calculate fitness variance
            scaling_factor = np.exp(-0.009 * fitness_variance)  # Modified scaling based on fitness variance

            for i in range(self.population_size):
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                a = best_solution
                donor = a + scaling_factor * np.random.uniform(0.4, 0.9) * (b - c)
                donor = np.clip(donor, lb, ub)

                if np.random.rand() < 0.5:
                    donor = self.quantum_inspired_mutation(population[i], best_solution, lb, ub)

                trial = self.differential_crossover(population[i], donor, cr=0.9)
                trial_fitness = func(trial)
                self.budget -= 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            if len(self.memory) < self.memory_size:
                self.memory.append((population, fitness))
            else:
                self.memory.pop(0)
                self.memory.append((population, fitness))

        best_index = np.argmin(fitness)
        return population[best_index]