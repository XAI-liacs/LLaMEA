import numpy as np

class QuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.best_pos = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.alpha = 0.1  # Adaptive mutation rate
        self.mutation_scale = 0.02  # Scale for mutation step size

    def mutation(self, individual):
        mutation_step = np.random.normal(scale=self.mutation_scale, size=self.dim)
        return np.clip(individual + mutation_step, 0, 1)

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def _evaluate_population(self, func):
        scores = np.array([func(ind * (func.bounds.ub - func.bounds.lb) + func.bounds.lb) for ind in self.positions])
        best_idx = np.argmin(scores)
        if scores[best_idx] < self.best_score:
            self.best_score = scores[best_idx]
            self.best_pos = self.positions[best_idx]
        self.evaluations += len(self.positions)
        return scores

    def _select_parents(self, scores):
        probabilities = 1 / (scores + 1e-9)
        probabilities /= probabilities.sum()
        parents_indices = np.random.choice(self.population_size, size=(self.population_size, 2), p=probabilities)
        return parents_indices

    def _adapt_mutation_scale(self):
        self.mutation_scale = min(0.1, self.mutation_scale + self.alpha)

    def __call__(self, func):
        while self.evaluations < self.budget:
            scores = self._evaluate_population(func)
            if self.evaluations >= self.budget:
                break

            parents_indices = self._select_parents(scores)
            new_population = []
            for parent1_idx, parent2_idx in parents_indices:
                parent1, parent2 = self.positions[parent1_idx], self.positions[parent2_idx]
                child1, child2 = self.crossover(parent1, parent2)
                if np.random.rand() < self.alpha:
                    child1 = self.mutation(child1)
                if np.random.rand() < self.alpha:
                    child2 = self.mutation(child2)
                new_population.extend([child1, child2])
                if len(new_population) >= self.population_size:
                    break

            self.positions = np.array(new_population[:self.population_size])
            self._adapt_mutation_scale()

        return self.best_pos * (func.bounds.ub - func.bounds.lb) + func.bounds.lb