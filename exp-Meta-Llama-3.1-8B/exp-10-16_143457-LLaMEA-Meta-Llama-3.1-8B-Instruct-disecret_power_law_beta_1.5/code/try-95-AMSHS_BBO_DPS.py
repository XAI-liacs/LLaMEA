import numpy as np
import random

class AMSHS_BBO_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.swarm_size = 10
        self.harmony_memory_size = 10
        self.paranoid = 0.1
        self.bw = 0.1
        self.hsr = 0.1
        self.rand = random.Random()
        self.convergence_threshold = 0.01
        self.convergence_counter = 0
        self.max_convergence_counter = 10

    def _generate_initial_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _evaluate_function(self, x):
        if self.budget > 0:
            self.budget -= 1
            return func(x)
        else:
            raise Exception("Budget exceeded")

    def _harmony_search(self, x, bounds):
        for i in range(self.dim):
            if self.rand.random() < self.hsr:
                x[i] = self.rand.uniform(bounds[i, 0], bounds[i, 1])
        return x

    def _update_harmony_memory(self, x, memory):
        if self.rand.random() < self.paranoid:
            memory[self.rand.randint(0, self.harmony_memory_size - 1)] = x
        return memory

    def _adjust_parameters(self):
        if self.convergence_counter >= self.max_convergence_counter:
            self.paranoid *= 0.9
            self.bw *= 0.9
            self.hsr *= 0.9
            self.convergence_counter = 0
        if self.rand.random() < 0.5:
            self.population_size *= 1.1
            self.swarm_size *= 1.1
            self.harmony_memory_size *= 1.1
        if self.rand.random() < 0.5:
            self.paranoid *= 1.1
            self.bw *= 1.1
            self.hsr *= 1.1

    def __call__(self, func):
        population = self._generate_initial_population()
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        best_solution = np.zeros(self.dim)
        best_fitness = float('inf')
        convergence = float('inf')

        for i in range(self.budget):
            for j in range(self.population_size):
                x = population[j]
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                population[j] = x

            for k in range(self.swarm_size):
                x = self.rand.uniform(self.lower_bound, self.upper_bound, self.dim)
                x = self._harmony_search(x, np.array([[self.lower_bound, self.upper_bound]] * self.dim))
                fitness = self._evaluate_function(x)
                if fitness < best_fitness:
                    best_solution = x
                    best_fitness = fitness
                harmony_memory = self._update_harmony_memory(x, harmony_memory)

            convergence = np.mean([self._evaluate_function(x) for x in population])
            if abs(convergence - best_fitness) < self.convergence_threshold:
                self.convergence_counter += 1
            else:
                self.convergence_counter = 0

            self._adjust_parameters()

        return best_solution, best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 1000
dim = 10
amshs_bbo_dps = AMSHS_BBO_DPS(budget, dim)
best_solution, best_fitness = amshs_bbo_dps(func)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)