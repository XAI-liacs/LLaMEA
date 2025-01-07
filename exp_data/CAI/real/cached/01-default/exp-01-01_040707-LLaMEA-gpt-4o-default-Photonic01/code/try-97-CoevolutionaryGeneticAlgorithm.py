import numpy as np

class CoevolutionaryGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.niche_radius = 0.1 * (budget / (dim * self.population_size)) ** 0.5
        self.mutation_prob = 0.05
        self.crossover_rate = 0.8
        self.position = None
        self.scores = None

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.scores[i]:
                self.scores[i] = scores[i]
        return scores

    def select_parents(self):
        total_score = np.sum(1 / (1 + self.scores))
        probabilities = (1 / (1 + self.scores)) / total_score
        indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        return self.position[indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim)
            return np.concatenate((parent1[:point], parent2[point:]))
        else:
            return parent1 if np.random.rand() > 0.5 else parent2

    def mutate(self, individual):
        for j in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                individual[j] += np.random.normal(0, 0.1)
        return individual

    def dynamic_niche_partitioning(self, niche_radius):
        niches = []
        for i in range(self.population_size):
            new_niche = True
            for niche in niches:
                if np.linalg.norm(self.position[i] - niche) < niche_radius:
                    new_niche = False
                    break
            if new_niche:
                niches.append(self.position[i])
        return niches

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            parents = self.select_parents()
            new_population = []
            for i in range(0, self.population_size, 2):
                offspring1 = self.crossover(parents[i], parents[i + 1])
                offspring2 = self.crossover(parents[i + 1], parents[i])
                new_population.extend([self.mutate(offspring1), self.mutate(offspring2)])
            self.position = np.array(new_population)
            niches = self.dynamic_niche_partitioning(self.niche_radius)
            if len(niches) < self.population_size:
                self.position[:len(niches)] = niches

        best_index = np.argmin(self.scores)
        return self.position[best_index], self.scores[best_index]