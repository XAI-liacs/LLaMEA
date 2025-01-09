import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.crossover_rate = 0.7
        self.learning_rate = 0.1  # Initial learning rate
        self.population_size = 10
        self.population = None
        self.best_solution = None
        self.best_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        self.evaluate_population(func)

        while self.evaluations < self.budget:
            for _ in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    parent1, parent2 = self.select_parents()
                    offspring = self.crossover(parent1, parent2, lb, ub)
                else:
                    candidate = self.mutate(lb, ub)
                    offspring = self.local_search(func, candidate, lb, ub)

                offspring_value = func(offspring)
                self.evaluations += 1

                if offspring_value < self.best_value:
                    self.best_solution, self.best_value = offspring, offspring_value
                    self.crossover_rate = min(1.0, self.crossover_rate + 0.02)
                    self.learning_rate *= 1.1
                else:
                    self.crossover_rate = max(0.2, self.crossover_rate - 0.01)
                    self.learning_rate *= 0.9
                
                self.replace_worst(offspring, offspring_value)
                
        return self.best_solution, self.best_value

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_population(self, func):
        for individual in self.population:
            value = func(individual)
            self.evaluations += 1
            if value < self.best_value:
                self.best_solution, self.best_value = individual, value

    def select_parents(self):
        idx = np.random.choice(self.population_size, 2, replace=False)
        return self.population[idx]

    def crossover(self, parent1, parent2, lb, ub):
        alpha = np.random.uniform(0, 1, self.dim)
        offspring = alpha * parent1 + (1 - alpha) * parent2
        return np.clip(offspring, lb, ub)

    def mutate(self, lb, ub):
        mutation_scale = np.random.rand() * (0.3 * (1 - (self.evaluations / self.budget)**2))
        candidate = np.random.normal(self.best_solution, (ub - lb) / 10 * mutation_scale, self.dim)
        return np.clip(candidate, lb, ub)

    def local_search(self, func, solution, lb, ub, alpha=1e-2, beta=0.9):
        gradient = self.estimate_gradient(func, solution, lb, ub)
        step_size = self.learning_rate * np.linalg.norm(gradient) / (1 + self.evaluations / self.budget)
        
        while step_size > 1e-5:
            candidate = solution - step_size * gradient
            candidate = np.clip(candidate, lb, ub)
            if func(candidate) < func(solution):
                return candidate
            step_size *= beta
            self.evaluations += 1
        return solution

    def estimate_gradient(self, func, solution, lb, ub, epsilon=1e-4):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            perturb[i] = epsilon
            upper_sol = np.clip(solution + perturb, lb, ub)
            lower_sol = np.clip(solution - perturb, lb, ub)
            gradient[i] = (func(upper_sol) - func(lower_sol)) / (2 * epsilon)
            self.evaluations += 2
        return gradient

    def replace_worst(self, candidate, candidate_value):
        worst_idx = np.argmax([func(x) for x in self.population])
        if candidate_value < func(self.population[worst_idx]):
            self.population[worst_idx] = candidate