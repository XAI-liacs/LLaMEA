import numpy as np
from scipy.optimize import minimize

class AdaptiveMemeticDEPA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.base_mutation_factor = 0.5
        self.crossover_rate = 0.85
        self.periodicity_factor = 0.15
        self.local_search_prob = 0.3
        self.best_solution = None
        self.diversity_threshold = 0.01

    def periodicity_cost(self, solution):
        periodic_error = np.var(solution[::2] - solution[1::2]) + np.var(solution - np.roll(solution, 3))
        return periodic_error

    def initialize_population(self, lb, ub):
        return lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)

    def adapt_mutation_factor(self, generation, diversity):
        factor_adjustment = 0.1 * (1 - diversity)
        return self.base_mutation_factor + factor_adjustment * np.sin(2 * np.pi * generation / 12)

    def mutate(self, population, idx, generation, diversity):
        indices = np.random.choice(range(self.pop_size), 3, replace=False)
        while idx in indices:
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
        a, b, c = population[indices]
        mutation_factor = self.adapt_mutation_factor(generation, diversity)
        mutant = a + mutation_factor * (b - c) + 0.10 * (np.random.rand(self.dim) - 0.5)  # Changed line
        return np.clip(mutant, 0, 1)

    def crossover(self, target, mutant, generation):
        dynamic_crossover_rate = 0.5 + 0.35 * np.cos(2 * np.pi * generation / 20)
        cross_points = np.random.rand(self.dim) < dynamic_crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target, trial, func):
        target_cost = func(target) + self.periodicity_factor * self.periodicity_cost(target)
        trial_cost = func(trial) + self.periodicity_factor * self.periodicity_cost(trial)
        return trial if trial_cost < target_cost else target

    def local_refinement(self, solution, func, lb, ub):
        res = minimize(func, solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return res.x

    def calculate_diversity(self, population):
        mean_sol = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - mean_sol, axis=1))
        return diversity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0
        generation = 0

        while evaluations < self.budget:
            new_population = []
            diversity = self.calculate_diversity(population)
            for idx in range(self.pop_size):
                target = population[idx]
                mutant = self.mutate(population, idx, generation, diversity)
                trial = self.crossover(target, mutant, generation)
                trial = lb + trial * (ub - lb)
                selected = self.select(target, trial, func)
                if np.random.rand() < 0.2 + 0.1 * diversity:  # Changed line
                    selected = self.local_refinement(selected, func, lb, ub)
                new_population.append(selected)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.array(new_population)
            generation += 1

        best_idx = np.argmin([func(ind) + (self.periodicity_factor - 0.05 * np.sin(2 * np.pi * generation / 20)) * self.periodicity_cost(ind) for ind in population])  # Changed line
        best_solution = population[best_idx]

        if self.best_solution is None or func(best_solution) < func(self.best_solution):
            self.best_solution = best_solution

        # Updated a single line by adding generation-based dynamic periodicity factor adjustment
        result = minimize(lambda x: func(x) + (self.periodicity_factor + 0.1 * np.cos(2 * np.pi * generation / 30)) * self.periodicity_cost(x),  # Changed line
                          self.best_solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return result.x