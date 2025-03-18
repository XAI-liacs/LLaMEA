import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class AdaptivePeriodicOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.fitness = np.inf * np.ones(self.population_size)
        self.bounds = None

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def periodic_guided_mutation(self, individual, lb, ub, t):
        periodicity = np.sin(2 * np.pi * t / self.budget)
        scale_factor = np.random.uniform(0.4, 0.9)  
        perturbation = scale_factor * periodicity * (ub - lb)
        return np.clip(individual + perturbation, lb, ub)

    def evaluate_population(self, pop):
        return np.array([self.func(ind) for ind in pop])

    def select_best(self, pop, fitness):
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]

    def bayesian_local_refinement(self, candidate, lb, ub):
        def acquisition(x):
            x = np.atleast_2d(x)
            mu, sigma = self.predict(x)
            return mu - 1.96 * sigma

        bounds = list(zip(lb, ub))
        res = minimize(acquisition, candidate, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget // 20})
        return res.x, self.func(res.x)

    def predict(self, x):
        distances = np.linalg.norm(x - self.pop, axis=1)
        weights = norm.pdf(distances)
        weights /= np.sum(weights)
        mu = np.dot(weights, self.fitness)
        sigma = np.sqrt(np.dot(weights, (self.fitness - mu) ** 2))
        return mu, sigma

    def __call__(self, func):
        self.func = func
        self.bounds = func.bounds
        lb, ub = self.bounds.lb, self.bounds.ub
        
        self.pop = self.initialize_population(lb, ub)
        self.fitness = self.evaluate_population(self.pop)
        evaluations = len(self.pop)

        while evaluations < self.budget:
            for i in range(self.population_size):
                candidate = self.periodic_guided_mutation(self.pop[i], lb, ub, evaluations)
                candidate_fitness = self.func(candidate)
                if candidate_fitness < self.fitness[i]:
                    self.pop[i], self.fitness[i] = candidate, candidate_fitness
                evaluations += 1

            if evaluations < self.budget:
                best_solution, best_fitness = self.select_best(self.pop, self.fitness)
                refined_solution, refined_fitness = self.bayesian_local_refinement(best_solution, lb, ub)
                if refined_fitness < best_fitness:
                    best_solution, best_fitness = refined_solution, refined_fitness
                evaluations += self.budget // 20

        return best_solution