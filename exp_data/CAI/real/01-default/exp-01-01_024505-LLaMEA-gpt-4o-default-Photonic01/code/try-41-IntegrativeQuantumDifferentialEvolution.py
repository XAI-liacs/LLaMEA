import numpy as np

class IntegrativeQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.quantum_weight = 0.6  # Quantum variability factor
        self.generational_progress = 0

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.best_fitness:
            self.best_fitness = fitness[best_index]
            self.best_solution = self.population[best_index]
        return fitness

    def select_mutant(self, target_idx, lb, ub):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = np.clip(self.population[a] + self.f * (self.population[b] - self.population[c]), lb, ub)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def apply_quantum_variation(self, individual, lb, ub):
        if np.random.rand() < self.quantum_weight:
            quantum_shift = np.random.normal(0, 0.1, self.dim)
            individual += quantum_shift
            individual = np.clip(individual, lb, ub)
        return individual

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += len(fitness)

            if evaluations >= self.budget:
                break

            new_population = np.copy(self.population)
            for i in range(self.population_size):
                mutant = self.select_mutant(i, lb, ub)
                offspring = self.crossover(self.population[i], mutant)
                offspring = self.apply_quantum_variation(offspring, lb, ub)
                if func(offspring) < fitness[i]:
                    new_population[i] = offspring

            self.population = new_population
            self.generational_progress += 1

        return self.best_solution, self.best_fitness