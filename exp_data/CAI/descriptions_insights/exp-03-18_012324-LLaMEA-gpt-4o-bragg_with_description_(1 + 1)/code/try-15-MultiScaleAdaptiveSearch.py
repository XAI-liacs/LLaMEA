import numpy as np
from concurrent.futures import ThreadPoolExecutor

class MultiScaleAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def differential_evolution(self, bounds, population_size, generations, func):
        def ensure_bounds(vec, bounds):
            return np.clip(vec, bounds.lb, bounds.ub)

        def mutation(population, bounds):
            indices = np.random.choice(range(population_size), 3, replace=False)
            a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
            F = np.random.uniform(0.5, 1.0)  # Adaptive mutation rate
            mutant = ensure_bounds(a + F * (b - c), bounds)
            return mutant

        population = np.random.rand(population_size, self.dim)
        for i in range(population_size):
            population[i] = bounds.lb + population[i] * (bounds.ub - bounds.lb)
        best_solution = None
        best_score = float('inf')

        for g in range(generations):
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(population_size):
                    futures.append(executor.submit(self.evaluate_trial, population, i, bounds, func, mutation))
                for future in futures:
                    trial, score, i = future.result()
                    if score < best_score:
                        best_score = score
                        best_solution = trial
                    if score < func(population[i]):
                        population[i] = trial

            if len(population) > 4 * self.dim:
                population_size = max(5, population_size//2)
                population = population[:population_size]
        return best_solution, best_score

    def evaluate_trial(self, population, i, bounds, func, mutation):
        mutant = mutation(population, bounds)
        cross_points = np.random.rand(self.dim) < 0.9
        trial = np.where(cross_points, mutant, population[i])
        score = func(trial)
        return trial, score, i

    def simulated_annealing(self, initial_solution, initial_temp, cooling_rate, bounds, func):
        current_solution = initial_solution
        current_score = func(current_solution)
        best_solution = np.copy(current_solution)
        best_score = current_score
        temp = initial_temp

        while temp > 1e-6:
            candidate_solution = current_solution + np.random.normal(0, temp, self.dim)
            candidate_solution = np.clip(candidate_solution, bounds.lb, bounds.ub)
            candidate_score = func(candidate_solution)
            if candidate_score < best_score:
                best_score = candidate_score
                best_solution = candidate_solution
            acceptance_prob = np.exp((current_score - candidate_score) / temp)
            if candidate_score < current_score or np.random.rand() < acceptance_prob:
                current_solution = candidate_solution
                current_score = candidate_score
            temp *= 0.95  # Adjusted cooling rate
        return best_solution, best_score

    def __call__(self, func):
        bounds = func.bounds
        population_size = 10 * self.dim
        generations = self.budget // population_size
        best_solution, _ = self.differential_evolution(bounds, population_size, generations, func)
        initial_temp = 1.0
        cooling_rate = 0.9
        best_solution, best_score = self.simulated_annealing(best_solution, initial_temp, cooling_rate, bounds, func)
        return best_solution