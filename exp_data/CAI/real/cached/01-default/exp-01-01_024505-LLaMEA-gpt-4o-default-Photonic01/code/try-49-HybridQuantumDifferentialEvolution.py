import numpy as np

class HybridQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.fitness = None
        self.best_position = None
        self.best_fitness = float('inf')
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.interference_prob = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate_population(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.population[i]
        
    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_vector = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < self.crossover_rate:
                crossover_vector[i] = mutant[i]
        return crossover_vector

    def apply_quantum_interference(self, individual, lb, ub):
        if np.random.rand() < self.interference_prob:
            interference_vector = lb + (ub - lb) * np.random.rand(self.dim)
            return np.mean([individual, interference_vector], axis=0)
        return individual

    def local_search(self, candidate, lb, ub):
        candidate += np.random.uniform(-0.1, 0.1, self.dim)
        return np.clip(candidate, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = self.apply_quantum_interference(trial, lb, ub)
                trial = np.clip(trial, lb, ub)

                trial_fitness = func(trial)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_position = trial

                self.population[i] = self.local_search(self.population[i], lb, ub)

        return self.best_position, self.best_fitness