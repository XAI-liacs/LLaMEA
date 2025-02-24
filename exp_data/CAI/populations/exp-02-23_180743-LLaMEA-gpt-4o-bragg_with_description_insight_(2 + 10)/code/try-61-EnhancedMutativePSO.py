import numpy as np
from scipy.optimize import minimize

class EnhancedMutativePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(5, 10 * dim // 2)
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive parameter
        self.c2 = 1.5  # social parameter
        self.bounds = None
        self.F_min, self.F_max = 0.4, 0.9  # Adaptive mutation factors
        self.V_max = None

    def initialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        self.V_max = (ub - lb) * 0.1
        return np.random.uniform(lb, ub, (self.pop_size, self.dim)), np.zeros((self.pop_size, self.dim))

    def mutate_particle(self, particle):
        F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
        mutation = F * (np.random.rand(self.dim) - 0.5)
        return np.clip(particle + mutation, self.bounds.lb, self.bounds.ub)

    def apply_periodicity(self, solution):
        period = max(1, self.dim // 4)
        return np.tile(solution[:period], self.dim // period)

    def local_optimize(self, best_solution, func):
        result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(self.bounds.lb, self.bounds.ub)))
        return result.x if result.success else best_solution

    def update_velocity(self, velocity, personal_best, global_best, position):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive = self.c1 * r1 * (personal_best - position)
        social = self.c2 * r2 * (global_best - position)
        new_velocity = self.w * velocity + cognitive + social
        return np.clip(new_velocity, -self.V_max, self.V_max)

    def __call__(self, func):
        self.bounds = func.bounds
        positions, velocities = self.initialize_population()
        personal_best = positions.copy()
        personal_best_fitness = np.array([func(ind) for ind in positions])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        self.budget -= self.pop_size

        while self.budget > 0:
            for i in range(self.pop_size):
                velocities[i] = self.update_velocity(velocities[i], personal_best[i], global_best, positions[i])
                positions[i] = np.clip(positions[i] + velocities[i], self.bounds.lb, self.bounds.ub)
                positions[i] = self.mutate_particle(positions[i])
                positions[i] = self.apply_periodicity(positions[i])
                current_fitness = func(positions[i])

                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = positions[i]
                    personal_best_fitness[i] = current_fitness

                if current_fitness < global_best_fitness:
                    global_best = positions[i]
                    global_best_fitness = current_fitness

                self.budget -= 1
                if self.budget <= 0:
                    break

            if self.budget > 0:
                global_best = self.local_optimize(global_best, func)
                global_best_fitness = func(global_best)
                self.budget -= 1

        return global_best