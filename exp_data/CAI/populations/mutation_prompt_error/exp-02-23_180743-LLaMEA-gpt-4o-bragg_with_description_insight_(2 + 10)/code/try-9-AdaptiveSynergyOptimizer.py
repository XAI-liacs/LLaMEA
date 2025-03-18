import numpy as np
from scipy.optimize import minimize

class AdaptiveSynergyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.inertia_weight = 0.5
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        return np.random.rand(self.population_size, self.dim) * (ub - lb) + lb

    def differential_evolution(self, func, bounds, population, fitness):
        lb, ub = bounds.lb, bounds.ub
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.mutation_factor * (x1 - x2), lb, ub)

                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

        return population, fitness

    def particle_swarm_optimization(self, func, bounds, population, fitness):
        lb, ub = bounds.lb, bounds.ub
        velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) * 0.1
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best_positions[i] - population[i])
                    + self.social_constant * r2 * (global_best_position - population[i])
                )
                population[i] = np.clip(population[i] + velocities[i], lb, ub)
                f_value = func(population[i])
                self.evaluations += 1

                if f_value < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = f_value
                    if f_value < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best_position = personal_best_positions[i]

        return global_best_position

    def local_optimization(self, solution, func, bounds):
        result = minimize(func, solution, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        return result.x if result.success else solution

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds.lb, bounds.ub)
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size

        population, fitness = self.differential_evolution(func, bounds, population, fitness)
        best_solution = self.particle_swarm_optimization(func, bounds, population, fitness)
        best_solution = self.local_optimization(best_solution, func, bounds)
        return best_solution