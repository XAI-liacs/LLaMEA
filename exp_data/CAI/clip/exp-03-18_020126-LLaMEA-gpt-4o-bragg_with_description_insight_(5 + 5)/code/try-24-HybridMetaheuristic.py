import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.bounds = None

    def initialize_population(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = lb + ub - pop  # Quasi-Oppositional initialization
        return np.vstack((pop, opp_pop))

    def evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def select_best(self, population, fitness):
        idx = np.argmin(fitness)
        return population[idx], fitness[idx]

    def enforce_periodicity(self, candidate):
        # Adjust layers to encourage periodic solutions
        half_dim = self.dim // 2
        candidate[:half_dim] = candidate[half_dim:] = np.mean(candidate.reshape(-1, 2), axis=1)
        return candidate

    def crossover_and_mutate(self, target, mutant, gen_count, cr_start=0.9, cr_end=0.5):
        cr = cr_start + (cr_end - cr_start) * (gen_count / (self.budget / self.population_size))
        cross_points = np.random.rand(self.dim) < cr
        offspring = np.where(cross_points, mutant, target)
        return self.enforce_periodicity(offspring)

    def differential_evolution(self, func):
        lb, ub = self.bounds.lb, self.bounds.ub
        population = self.initialize_population(lb, ub)
        fitness = self.evaluate_population(population, func)
        eval_count = len(population)

        gen_count = 0  # Added line for generation count

        while eval_count < self.budget:
            for i in range(len(population)):
                if eval_count >= self.budget:
                    break
                indices = [idx for idx in range(len(population)) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), lb, ub)
                trial = self.crossover_and_mutate(population[i], mutant, gen_count)
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness

            gen_count += 1  # Added line to increment generation count

        return self.select_best(population, fitness)

    def local_optimization(self, best_solution, func):
        result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return result.x if result.success else best_solution

    def __call__(self, func):
        self.bounds = func.bounds
        best_solution, _ = self.differential_evolution(func)
        best_solution = self.local_optimization(best_solution, func)
        return best_solution