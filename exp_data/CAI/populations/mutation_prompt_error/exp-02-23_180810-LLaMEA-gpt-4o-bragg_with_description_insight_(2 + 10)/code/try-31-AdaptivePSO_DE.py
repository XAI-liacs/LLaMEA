import numpy as np
from scipy.optimize import minimize

class AdaptivePSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pso_w = 0.5
        self.pso_c1 = 1.5
        self.pso_c2 = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.evaluations = 0

    def periodic_enforced_initialization(self, bounds):
        lb, ub = bounds
        mid_point = (ub + lb) / 2
        span = (ub - lb) / 2
        base_pop = np.random.rand(self.population_size, self.dim // 2)
        pop = np.tile(base_pop, 2)
        return mid_point + span * (2 * pop - 1)

    def adaptive_pso(self, func, bounds):
        lb, ub = bounds
        population = self.periodic_enforced_initialization(bounds)
        velocity = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        while self.evaluations < self.budget // 2:
            for i in range(self.population_size):
                if self.evaluations >= self.budget // 2:
                    break

                # Update velocity and position
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.pso_w * velocity[i]
                               + self.pso_c1 * r1 * (personal_best[i] - population[i])
                               + self.pso_c2 * r2 * (global_best - population[i]))
                population[i] = np.clip(population[i] + velocity[i], lb, ub)

                # Evaluation
                score = func(population[i])
                self.evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best[i] = population[i]
                    personal_best_scores[i] = score

                    # Update global best
                    if score < global_best_score:
                        global_best, global_best_score = population[i], score

        return global_best

    def differential_evolution(self, func, bounds, initial_solution):
        lb, ub = bounds
        population = self.periodic_enforced_initialization(bounds)
        population[0] = initial_solution
        best_solution = initial_solution
        best_score = func(best_solution)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                self.mutation_factor = 0.5 + 0.3 * np.random.rand()  # Adaptive mutation factor
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                # Crossover
                self.crossover_rate = 0.8 + 0.2 * np.random.rand()  # Adaptive crossover rate
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Evaluation
                score = func(trial)
                self.evaluations += 1

                # Selection
                if score < func(population[i]):
                    population[i] = trial
                    if score < best_score:
                        best_solution, best_score = trial, score

        return best_solution

    def local_search(self, func, best_solution, bounds):
        res = minimize(lambda x: func(x), best_solution, bounds=bounds, method='L-BFGS-B')
        return res.x

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        global_best = self.adaptive_pso(func, bounds)
        best_solution = self.differential_evolution(func, bounds, global_best)
        best_solution = self.local_search(func, best_solution, bounds)
        return best_solution