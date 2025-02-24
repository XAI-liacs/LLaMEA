import numpy as np
from scipy.optimize import minimize

class HybridPSODynamicPeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.used_evaluations = 0
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

    def _initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        positions = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        velocities = np.zeros((self.population_size, self.dim))
        return positions, velocities

    def _evaluate_particles(self, func, positions):
        fitness = np.zeros(len(positions))
        for i in range(len(positions)):
            if self.used_evaluations < self.budget:
                fitness[i] = func(positions[i])
                self.used_evaluations += 1
        return fitness

    def _update_velocities_and_positions(self, positions, velocities, personal_best_positions, global_best_position, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities[i] = (self.inertia_weight * velocities[i] +
                             self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                             self.social_coeff * r2 * (global_best_position - positions[i]))
            velocities[i] = np.clip(velocities[i], lb - positions[i], ub - positions[i])
            positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
        return positions, velocities

    def _dynamic_periodicity_adjustment(self, positions):
        periodicity_factor = np.cos(self.used_evaluations * np.pi / self.budget) + 1
        for i in range(len(positions)):
            positions[i] += (np.mean(positions) - positions[i]) * periodicity_factor * 0.01
        return positions

    def _local_fine_tuning(self, best_solution, func, bounds):
        if self.used_evaluations >= self.budget:
            return best_solution
        result = minimize(func, best_solution, bounds=list(zip(bounds.lb, bounds.ub)),
                          method='L-BFGS-B', options={'maxfun': self.budget - self.used_evaluations})
        self.used_evaluations += result.nfev
        return result.x if result.success else best_solution

    def __call__(self, func):
        bounds = func.bounds
        positions, velocities = self._initialize_particles(bounds)
        fitness = self._evaluate_particles(func, positions)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]

        while self.used_evaluations < self.budget:
            positions = self._dynamic_periodicity_adjustment(positions)
            positions, velocities = self._update_velocities_and_positions(
                positions, velocities, personal_best_positions, global_best_position, bounds)
            fitness = self._evaluate_particles(func, positions)
            for i in range(self.population_size):
                if fitness[i] < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness[i]
            new_global_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[new_global_best_idx] < personal_best_fitness[global_best_idx]:
                global_best_idx = new_global_best_idx
                global_best_position = personal_best_positions[global_best_idx]

        best_solution = self._local_fine_tuning(global_best_position, func, bounds)
        
        return best_solution