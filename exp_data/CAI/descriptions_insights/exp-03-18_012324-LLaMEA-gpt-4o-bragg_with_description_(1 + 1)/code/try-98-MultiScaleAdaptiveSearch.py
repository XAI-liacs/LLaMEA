import numpy as np

class MultiScaleAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def differential_evolution(self, bounds, population_size, generations, func):
        def ensure_bounds(vec, bounds):
            return np.clip(vec, bounds.lb, bounds.ub)

        population = np.random.rand(population_size, self.dim)
        for i in range(population_size):
            population[i] = bounds.lb + population[i] * (bounds.ub - bounds.lb)
        best_solution = None
        best_score = float('inf')

        for g in range(generations):
            adaptive_factor = 0.6 + 0.4 * np.cos(np.pi * g / generations)  # Changed adaptive factor
            for i in range(population_size):
                indices = np.random.choice(range(population_size), 3, replace=False)
                a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
                mutant = ensure_bounds(a + np.random.rand() * adaptive_factor * (b - c), bounds)  # Modified mutation
                cross_points = np.random.rand(self.dim) < (0.8 + 0.2 * adaptive_factor)  # Adjusted crossover rate
                trial = np.where(cross_points, mutant, population[i])
                score = func(trial)
                if score < best_score:
                    best_score = score
                    best_solution = trial
                if score < func(population[i]):
                    population[i] = trial
            # Changed line: added adaptive population size adjustment
            if len(population) > 5 * self.dim:
                population_size = max(5, int(population_size * 0.9))  # Adjusted population shrink
                population = population[:population_size]
        best_solution = self.local_search(best_solution, bounds, func)
        return best_solution, best_score

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
            # Changed line: adjusted cooling multiplier for better exploration
            temp *= (0.94 + 0.06 * np.tanh(best_score))
        return best_solution, best_score

    def local_search(self, solution, bounds, func):
        perturbation = np.random.normal(0, 0.03, self.dim)  # Changed uniform to normal for refined search
        refined_solution = np.clip(solution + perturbation, bounds.lb, bounds.ub)
        return refined_solution if func(refined_solution) < func(solution) else solution

    def __call__(self, func):
        bounds = func.bounds
        population_size = 10 * self.dim
        generations = self.budget // population_size
        best_solution, _ = self.differential_evolution(bounds, population_size, generations, func)
        initial_temp = 1.0
        cooling_rate = 0.9
        best_solution, best_score = self.simulated_annealing(best_solution, initial_temp, cooling_rate, bounds, func)
        return best_solution