import numpy as np

class AMGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget)
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.elitism_count = 2
        self.local_search_prob = 0.3
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                mutation_value = np.random.uniform(-0.1, 0.1)
                individual[i] += mutation_value
                lb, ub = self.bounds
                individual[i] = np.clip(individual[i], lb[i], ub[i])
        return individual

    def local_search(self, individual):
        perturbation = np.random.uniform(-0.05, 0.05, self.dim)
        new_individual = individual + perturbation
        lb, ub = self.bounds
        return np.clip(new_individual, lb, ub)

    def select_parents(self):
        fitness_sum = np.sum(1.0 / (self.fitness + 1e-9))
        selection_probs = (1.0 / (self.fitness + 1e-9)) / fitness_sum
        parents_indices = np.random.choice(self.population_size, size=2, p=selection_probs)
        return self.population[parents_indices[0]], self.population[parents_indices[1]]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            new_population = []
            new_fitness = []

            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                if np.random.rand() < self.local_search_prob:
                    child1 = self.local_search(child1)
                if np.random.rand() < self.local_search_prob:
                    child2 = self.local_search(child2)

                new_population.extend([child1, child2])

                if evaluations < self.budget:
                    fitness1 = func(child1)
                    new_fitness.append(fitness1)
                    evaluations += 1
                if evaluations < self.budget:
                    fitness2 = func(child2)
                    new_fitness.append(fitness2)
                    evaluations += 1

            # Retain the best solutions (elitism)
            self.fitness = np.array(new_fitness)
            sorted_indices = np.argsort(self.fitness)
            new_population = np.array(new_population)
            new_population = new_population[sorted_indices]
            self.population = new_population[:self.population_size]
            self.fitness = self.fitness[sorted_indices][:self.population_size]

            # Adapt crossover and mutation rates based on diversity
            diversity = np.mean(np.std(self.population, axis=0))
            self.crossover_rate = 0.5 + 0.5 * (diversity / np.max(diversity, initial=1e-9))
            self.mutation_rate = 0.05 + 0.05 * (1 - diversity / np.max(diversity, initial=1e-9))

        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]