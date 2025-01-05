import numpy as np

class BioInspiredMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.local_search_probability = 0.3
        self.mutation_step_size = 0.05

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.dim)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            return child
        return parent1

    def mutate(self, individual, bounds):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.normal(0, self.mutation_step_size, self.dim)
            mutated_individual = individual + mutation_vector
            return np.clip(mutated_individual, bounds[:, 0], bounds[:, 1])
        return individual

    def local_search(self, individual, func, bounds):
        step_size = self.mutation_step_size * np.random.rand(self.dim)
        candidate = individual + step_size
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        if func(candidate) < func(individual):
            return candidate
        return individual

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                parents_indices = np.random.choice(self.population_size, 2, replace=False)
                parent1, parent2 = population[parents_indices]
                
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)

                if np.random.rand() < self.local_search_probability:
                    child1 = self.local_search(child1, func, bounds)
                    child2 = self.local_search(child2, func, bounds)

                new_population.extend([child1, child2])

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])
            eval_count += self.population_size

            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            best_value = fitness[best_idx]

        return best_individual