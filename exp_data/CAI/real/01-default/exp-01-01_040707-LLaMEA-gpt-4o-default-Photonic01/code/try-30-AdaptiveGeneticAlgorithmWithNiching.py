import numpy as np

class AdaptiveGeneticAlgorithmWithNiching:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.2
        self.crossover_rate = 0.7
        self.elite_archive_size = 10
        self.elite_archive = []
        self.niche_radius = None

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))
        self.elite_archive = []

    def evaluate(self, func):
        scores = np.array([func(ind) for ind in self.population])
        return scores

    def niche_count(self, candidate):
        return sum(np.linalg.norm(candidate - e) < self.niche_radius for e in self.elite_archive)

    def select_parents(self):
        niche_counts = np.array([self.niche_count(ind) for ind in self.population])
        fitness = 1 / (1 + self.scores + niche_counts)
        selected_indices = np.random.choice(self.population_size, size=self.population_size // 2, p=fitness/fitness.sum())
        return self.population[selected_indices]

    def crossover(self, parents):
        offspring = []
        for _ in range(len(parents) // 2):
            if np.random.rand() < self.crossover_rate:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                crossover_point = np.random.randint(1, self.dim)
                child1 = np.hstack((parents[p1][:crossover_point], parents[p2][crossover_point:]))
                child2 = np.hstack((parents[p2][:crossover_point], parents[p1][crossover_point:]))
                offspring.extend([child1, child2])
            else:
                offspring.extend(parents[np.random.choice(len(parents), 2, replace=False)])
        return np.array(offspring)

    def mutate(self, offspring):
        mutation_matrix = np.random.rand(*offspring.shape) < self.mutation_rate
        mutation_values = (np.random.rand(*offspring.shape) - 0.5) * 0.1
        offspring[mutation_matrix] += mutation_values[mutation_matrix]
        return offspring

    def update_elite_archive(self, scores):
        elite_candidates = list(zip(self.population, scores))
        elite_candidates.sort(key=lambda x: x[1])
        self.elite_archive.extend(elite_candidates[:self.elite_archive_size])
        self.elite_archive = sorted(self.elite_archive, key=lambda x: x[1])[:self.elite_archive_size]

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        self.niche_radius = 0.1 * (func.bounds.ub - func.bounds.lb).mean()

        while func_calls < self.budget:
            self.scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_elite_archive(self.scores)
            parents = self.select_parents()
            offspring = self.crossover(parents)
            self.population = self.mutate(offspring)
            iteration += 1

        best_individual = min(self.elite_archive, key=lambda x: x[1])
        return best_individual[0], best_individual[1]