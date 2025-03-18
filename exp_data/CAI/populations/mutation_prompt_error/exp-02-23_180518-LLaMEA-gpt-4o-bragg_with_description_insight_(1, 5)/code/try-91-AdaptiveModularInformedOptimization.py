import numpy as np
from scipy.optimize import minimize

class AdaptiveModularInformedOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(100, 10 * dim))
        self.pop = None
        self.velocities = None
        self.p_best = None
        self.p_best_val = np.full(self.population_size, np.inf)
        self.g_best = None
        self.g_best_val = np.inf
        self.evaluations = 0
        self.base_omega = 0.5
        self.phi_p = 0.5
        self.phi_g = 0.5

    def initialize_population(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-(ub-lb), (ub-lb), (self.population_size, self.dim))
        self.p_best = self.pop.copy()

    def modular_pattern_recognition(self, solution, lb, ub):
        segment_length = self.dim // 5
        for start in range(0, self.dim, segment_length):
            end = min(start + segment_length, self.dim)
            avg_value = np.mean(solution[start:end])
            solution[start:end] = avg_value + 0.1 * (ub - lb) * np.random.randn(end - start)
        return solution

    def adaptive_particle_swarm(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        fitness = np.apply_along_axis(func, 1, self.pop)
        self.evaluations += self.population_size
        better_mask = fitness < self.p_best_val
        self.p_best_val[better_mask] = fitness[better_mask]
        self.p_best[better_mask] = self.pop[better_mask]

        if fitness.min() < self.g_best_val:
            self.g_best_val = fitness.min()
            self.g_best = self.pop[fitness.argmin()].copy()

        while self.evaluations < self.budget:
            diversity = np.std(self.pop, axis=0).mean()
            self.base_omega = max(0.3, min(0.9, self.base_omega * (1 + 0.5 * (1 - diversity))))
            self.phi_g = max(0.4, min(1.0, self.phi_g + 0.1 * diversity))

            for i in range(self.population_size):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)
                self.velocities[i] = (self.base_omega * self.velocities[i] +
                                      self.phi_p * r_p * (self.p_best[i] - self.pop[i]) +
                                      self.phi_g * r_g * (self.g_best - self.pop[i]))

            self.pop = np.clip(self.pop + self.velocities, lb, ub)
            self.pop += 0.01 * (ub - lb) * np.random.randn(self.population_size, self.dim)

            for i in range(self.population_size):
                self.pop[i] = self.modular_pattern_recognition(self.pop[i], lb, ub)

            fitness = np.apply_along_axis(func, 1, self.pop)
            self.evaluations += self.population_size

            better_mask = fitness < self.p_best_val
            self.p_best_val[better_mask] = fitness[better_mask]
            self.p_best[better_mask] = self.pop[better_mask]

            if fitness.min() < self.g_best_val:
                self.g_best_val = fitness.min()
                self.g_best = self.pop[fitness.argmin()].copy()

            if self.evaluations >= self.budget:
                break

    def local_refinement(self, func):
        if self.g_best is not None:
            result = minimize(func, self.g_best, method='L-BFGS-B', bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
            self.evaluations += result.nfev

            if result.fun < self.g_best_val:
                self.g_best_val = result.fun
                self.g_best = result.x

    def __call__(self, func):
        self.adaptive_particle_swarm(func)
        self.local_refinement(func)
        return self.g_best, self.g_best_val