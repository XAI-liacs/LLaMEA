import numpy as np
from scipy.optimize import minimize

class PeriodicReflectivityOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Population size for DE
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.num_generations = int(budget / self.population_size)
        self.local_refinement_threshold = 0.1 * budget  # Budget threshold to trigger local refinement

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = self.initialize_population(bounds)
        best_solution = None
        best_score = float('inf')
        evaluations = 0

        for generation in range(self.num_generations):
            population_scores = np.array([self.evaluate_individual(ind, func) for ind in population])
            evaluations += len(population)

            if generation > self.num_generations * 0.5:  # Encourage periodicity in later generations
                population_scores += self.periodicity_penalty(population)

            generation_best_score = np.min(population_scores)
            if generation_best_score < best_score:
                best_score = generation_best_score
                best_solution = population[np.argmin(population_scores)]

            if evaluations >= self.local_refinement_threshold:
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds)
                if result.fun < best_score:
                    best_score, best_solution = result.fun, result.x
                evaluations += result.nfev
                break

            next_population = self.evolve_population(population, population_scores, bounds)
            population = next_population

        return best_solution

    def initialize_population(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))

    def evaluate_individual(self, individual, func):
        return func(individual)

    def periodicity_penalty(self, population):
        penalties = np.zeros(len(population))
        for i, individual in enumerate(population):
            period = 2  # Assuming known optimal period
            penalty = np.sum((individual[i::period] - individual[i]) ** 2 for i in range(period))
            penalties[i] = penalty
        return penalties

    def evolve_population(self, population, scores, bounds):
        next_population = np.empty_like(population)
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
            cross_points = np.random.rand(self.dim) < self.crossover_probability
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            next_population[i] = np.where(cross_points, mutant, population[i])
        return next_population