import numpy as np

class BioInspiredMemeticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 15))
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.1
        self.crossover_rate = 0.6
        self.local_search_prob = 0.2

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate_population(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i]

    def select_parents(self):
        probabilities = 1 / (1 + self.fitness)
        probabilities /= probabilities.sum()
        parents_indices = np.random.choice(self.population_size, size=2, p=probabilities)
        return self.population[parents_indices[0]], self.population[parents_indices[1]]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1, parent2

    def mutate(self, individual, lb, ub):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        return np.clip(individual, lb, ub)

    def local_search(self, individual, lb, ub, func):
        local_budget = max(1, self.budget // (10 * self.population_size))
        best_local = individual.copy()
        best_local_fitness = func(individual)
        for _ in range(local_budget):
            perturbation = np.random.normal(0, 0.1, self.dim)
            candidate = best_local + perturbation
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = func(candidate)
            if candidate_fitness < best_local_fitness:
                best_local = candidate
                best_local_fitness = candidate_fitness
        return best_local

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations + self.population_size <= self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, lb, ub)
                child2 = self.mutate(child2, lb, ub)
                new_population.extend([child1, child2])

            if len(new_population) < self.population_size:
                new_population.append(self.mutate(self.select_parents()[0], lb, ub))

            self.population = np.array(new_population[:self.population_size])

            for i in range(self.population_size):
                if np.random.rand() < self.local_search_prob:
                    self.population[i] = self.local_search(self.population[i], lb, ub, func)

        return self.best_solution, self.best_fitness