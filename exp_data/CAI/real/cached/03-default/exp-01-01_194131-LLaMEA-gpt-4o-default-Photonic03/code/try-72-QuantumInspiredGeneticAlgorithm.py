import numpy as np

class QuantumInspiredGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.pbest = self.population.copy()
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.mutation_rate = 0.1
        self.adaptive_factor = 0.3

    def diversity_maintenance(self):
        mean_vector = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - mean_vector, axis=1))
        if diversity < 0.1:
            self.population += 0.1 * (np.random.rand(self.population_size, self.dim) - 0.5)

    def adaptive_crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        if self.gbest is not None:
            beta = np.random.uniform(0, self.adaptive_factor, self.dim)
            offspring = beta * self.gbest + (1 - beta) * (alpha * parent1 + (1 - alpha) * parent2)
        else:
            offspring = alpha * parent1 + (1 - alpha) * parent2
        return offspring

    def mutate(self, individual):
        mutation_vector = np.random.normal(0, 1, self.dim) * self.mutation_rate
        return np.clip(individual + mutation_vector, 0, 1)

    def _update_population(self, func):
        new_population = []
        scores = []
        for i in range(self.population_size):
            parent1, parent2 = self.population[np.random.choice(self.population_size, 2, replace=False)]
            offspring = self.adaptive_crossover(parent1, parent2)
            offspring = self.mutate(offspring)
            offspring = np.clip(func.bounds.lb + (func.bounds.ub - func.bounds.lb) * offspring, func.bounds.lb, func.bounds.ub)
            score = func(offspring)
            new_population.append(offspring)
            scores.append(score)
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break
        return new_population, scores

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.population[i])
            self.pbest[i] = self.population[i]
            self.pbest_scores[i] = score
            if score < self.gbest_score:
                self.gbest = self.population[i]
                self.gbest_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            self.diversity_maintenance()
            new_population, new_scores = self._update_population(func)
            for i in range(len(new_population)):
                if new_scores[i] < self.pbest_scores[i]:
                    self.pbest[i] = new_population[i]
                    self.pbest_scores[i] = new_scores[i]
                if new_scores[i] < self.gbest_score:
                    self.gbest = new_population[i]
                    self.gbest_score = new_scores[i]
            self.population = new_population

        return self.gbest