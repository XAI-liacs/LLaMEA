import numpy as np

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + int(0.5 * dim)  # Adjust population size based on dimension
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.shrink_factor = 0.9  # New variable to dynamically adjust mutation factor

    def opposition_based_initialization(self):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        opposition_population = self.bounds[0] + self.bounds[1] - population
        return np.vstack((population, opposition_population))

    def quantum_superposition(self, best, individual):
        alpha = np.random.uniform(0, 1, self.dim)
        return alpha * best + (1 - alpha) * individual

    def differential_mutation(self, population, index, best_index):
        indices = [i for i in range(population.shape[0]) if i != index and i != best_index]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        return a + self.mutation_factor * (b - c) * self.shrink_factor  # Adjust mutation factor

    def crossover(self, target, donor):
        mask = np.random.rand(self.dim) < self.crossover_prob
        return np.where(mask, donor, target)

    def adapt_population_size(self, evaluations):
        if evaluations % (0.25 * self.budget) == 0:  # Reduce population size at intervals
            self.population_size = max(10, int(self.population_size * 0.8))

    def __call__(self, func):
        population = self.opposition_based_initialization()
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = population.shape[0]
        
        best_index = np.argmin(fitness)
        best = population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            self.adapt_population_size(evaluations)  # Call adaptive population size
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                donor = self.differential_mutation(population, i, best_index)
                trial = self.crossover(population[i], donor)
                trial = np.clip(trial, self.bounds[0], self.bounds[1])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness
                        best_index = i

                # Quantum superposition step
                quantum_candidate = self.quantum_superposition(best, trial)
                quantum_candidate = np.clip(quantum_candidate, self.bounds[0], self.bounds[1])
                quantum_fitness = func(quantum_candidate)
                evaluations += 1

                if quantum_fitness < fitness[i]:
                    population[i] = quantum_candidate
                    fitness[i] = quantum_fitness

                    if quantum_fitness < best_fitness:
                        best = quantum_candidate
                        best_fitness = quantum_fitness
                        best_index = i

        return best, best_fitness