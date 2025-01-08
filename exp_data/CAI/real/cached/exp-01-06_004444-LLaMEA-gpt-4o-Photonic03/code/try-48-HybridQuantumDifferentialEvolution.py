import numpy as np

class HybridQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.secondary_population_size = int(0.2 * self.population_size)  # New line
        self.memory_size = 5
        self.memory = []

    def initialize_population(self, lb, ub):
        primary = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        secondary = lb + (ub - lb) * np.random.rand(self.secondary_population_size, self.dim)  # New line
        return primary, secondary  # Modified line

    def quantum_inspired_mutation(self, individual, best_solution, lb, ub):
        dynamic_factor = (self.budget / (self.budget + 5)) * 1.02  # Adjusted line for increased dynamic factor
        amplitude = np.abs(np.random.normal(0, dynamic_factor, self.dim))
        mutated = individual + amplitude * (best_solution - individual)
        return np.clip(mutated, lb, ub)

    def differential_crossover(self, target, donor, cr):
        mask = np.random.rand(self.dim) < cr
        trial = np.where(mask, donor, target)
        return trial

    def adapt_learning_from_secondary(self, primary, secondary, cr=0.7):  # New function
        for i in range(self.secondary_population_size):
            if np.random.rand() < cr:
                primary[i] = self.differential_crossover(primary[i], secondary[i % self.secondary_population_size], cr)
        return primary
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        primary_population, secondary_population = self.initialize_population(lb, ub)  # Modified line
        fitness = np.array([func(ind) for ind in primary_population])
        self.budget -= self.population_size

        secondary_fitness = np.array([func(ind) for ind in secondary_population])  # New line
        self.budget -= self.secondary_population_size  # New line

        while self.budget > 0:
            best_index = np.argmin(fitness)
            best_solution = primary_population[best_index]

            new_population = []
            new_fitness = []

            for i in range(self.population_size):
                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = primary_population[np.random.choice(idxs, 3, replace=False)]
                a = best_solution
                dynamic_scaling = np.exp(-0.008 * i)  # Adjusted line for slightly increased dynamic scaling
                donor = a + dynamic_scaling * np.random.uniform(0.4, 0.9) * (b - c)
                donor = np.clip(donor, lb, ub)

                if np.random.rand() < 0.5:
                    donor = self.quantum_inspired_mutation(primary_population[i], best_solution, lb, ub)

                trial = self.differential_crossover(primary_population[i], donor, cr=0.9)
                trial_fitness = func(trial)
                self.budget -= 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(primary_population[i])
                    new_fitness.append(fitness[i])

            primary_population = np.array(new_population)
            fitness = np.array(new_fitness)

            primary_population = self.adapt_learning_from_secondary(primary_population, secondary_population)  # New line

            if len(self.memory) < self.memory_size:
                self.memory.append((primary_population, fitness))
            else:
                self.memory.pop(0)
                self.memory.append((primary_population, fitness))

        best_index = np.argmin(fitness)
        return primary_population[best_index]