import numpy as np

class HybridEvoSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, dim))
        self.p_best = self.population.copy()
        self.g_best = None
        self.p_best_scores = np.full(self.population_size, np.inf)
        self.g_best_score = np.inf
        self.best_individual = None
        self.best_score = np.inf
        self.evals = 0

    def __call__(self, func):
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive component
        c2 = 1.5  # Social component
        
        while self.evals < self.budget:
            for i in range(self.population_size):
                score = func(self.population[i])
                self.evals += 1
                
                if score < self.p_best_scores[i]:
                    self.p_best_scores[i] = score
                    self.p_best[i] = self.population[i].copy()
                
                if score < self.g_best_score:
                    self.g_best_score = score
                    self.g_best = self.population[i].copy()

            for i in range(self.population_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                self.velocities[i] = (
                    w * self.velocities[i] +
                    c1 * r1 * (self.p_best[i] - self.population[i]) +
                    c2 * r2 * (self.g_best - self.population[i])
                )
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                
            # Evolutionary phase - simulated binary crossover and mutation
            if self.evals < 0.5 * self.budget:
                for i in range(0, self.population_size, 2):
                    if np.random.rand() < 0.9:  # Crossover probability
                        parent1, parent2 = self.population[i], self.population[i+1]
                        child1, child2 = self.simulated_binary_crossover(parent1, parent2)
                        self.population[i], self.population[i+1] = child1, child2

                for i in range(self.population_size):
                    if np.random.rand() < 0.1:  # Mutation probability
                        self.population[i] = self.polynomial_mutation(self.population[i])

        self.best_individual = self.g_best
        self.best_score = self.g_best_score
        return self.best_individual

    def simulated_binary_crossover(self, parent1, parent2, eta=10):
        child1, child2 = np.empty_like(parent1), np.empty_like(parent2)
        for j in range(self.dim):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child1[j] = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
            child2[j] = 0.5 * ((1 - beta) * parent1[j] + (1 + beta) * parent2[j])
        return np.clip(child1, -5.0, 5.0), np.clip(child2, -5.0, 5.0)

    def polynomial_mutation(self, individual, eta=20):
        mutant = np.copy(individual)
        for j in range(self.dim):
            if np.random.rand() < 1 / self.dim:
                delta = (2 * np.random.rand()) ** (1 / (eta + 1)) - 1
                mutant[j] += delta
        return np.clip(mutant, -5.0, 5.0)