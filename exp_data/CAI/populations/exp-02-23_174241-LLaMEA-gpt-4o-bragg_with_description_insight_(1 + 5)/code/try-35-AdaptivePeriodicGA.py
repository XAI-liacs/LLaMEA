import numpy as np
from scipy.optimize import minimize

class AdaptivePeriodicGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4, 10 * dim)
        self.bounds = None
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, population, func):
        scores = np.zeros(self.population_size)
        for i in range(self.population_size):
            scores[i] = func(population[i]) + self.periodicity_penalty(population[i])
        return scores

    def select_parents(self, scores):
        return np.random.choice(self.population_size, size=2, replace=False, p=(1/scores)/np.sum(1/scores))

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim)
            return np.concatenate((parent1[:point], parent2[point:]))
        return parent1

    def mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.sin(np.random.uniform(0, np.pi)) * (self.bounds.ub[i] - self.bounds.lb[i])
                individual[i] = np.clip(individual[i], self.bounds.lb[i], self.bounds.ub[i])
        return individual

    def periodicity_penalty(self, solution):
        period = self.dim // 2
        penalties = [(solution[i] - solution[i + period]) ** 2 for i in range(self.dim - period)]
        return np.sum(penalties)

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        eval_count = 0

        while eval_count < self.budget:
            scores = self.evaluate_population(population, func)
            new_population = np.zeros_like(population)

            for i in range(self.population_size):
                parent_indices = self.select_parents(scores)
                parent1, parent2 = population[parent_indices]
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population[i] = offspring

            population = new_population

            best_idx = np.argmin(scores)
            best_solution = population[best_idx]
            best_score = scores[best_idx]

            # Local optimization using BFGS if budget allows
            if eval_count < self.budget:
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=zip(self.bounds.lb, self.bounds.ub))
                eval_count += result.nfev
                if result.fun < best_score:
                    best_score = result.fun
                    best_solution = result.x

            eval_count += self.population_size

        return best_solution