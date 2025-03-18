import numpy as np

class Quantum_Inspired_Genetic_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.alpha = 0.5  # quantum rotation angle
        self.beta = 0.1   # mutation probability

    def quantum_rotation(self, individual, best, worst):
        rotation = np.random.uniform(-self.alpha, self.alpha, size=self.dim)
        new_position = individual + rotation * (best - worst)
        return new_position

    def mutate(self, individual, lb, ub):
        mutation = np.random.rand(self.dim) < self.beta
        noise = np.random.uniform(lb, ub, size=self.dim)
        return np.where(mutation, noise, individual)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])

        evaluations = self.population_size
        best_idx = scores.argmin()
        global_best_position = population[best_idx].copy()
        global_best_score = scores[best_idx]

        while evaluations < self.budget:
            best_idx = scores.argmin()
            worst_idx = scores.argmax()
            best_individual = population[best_idx]
            worst_individual = population[worst_idx]

            for i in range(self.population_size):
                new_position = self.quantum_rotation(population[i], best_individual, worst_individual)
                new_position = np.clip(new_position, lb, ub)
                new_position = self.mutate(new_position, lb, ub)

                new_score = func(new_position)
                evaluations += 1

                if new_score < scores[i]:
                    population[i] = new_position
                    scores[i] = new_score
                    if new_score < global_best_score:
                        global_best_position = new_position.copy()
                        global_best_score = new_score
                
                if evaluations >= self.budget:
                    break
        
        return global_best_position, global_best_score