import numpy as np
from scipy.optimize import minimize

class MemeticMultiLayeredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pop = None
        self.velocities = None
        self.p_best = None
        self.p_best_val = None
        self.g_best = None
        self.g_best_val = np.inf
        self.evaluations = 0
        self.phi_p = 0.5  # Personal attraction coefficient
        self.phi_g = 0.5  # Global attraction coefficient

    def adaptive_inertia_weight(self):
        return 0.9 - (0.5 * (self.evaluations / self.budget))

    def initialize_population(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-(ub-lb), (ub-lb), (self.population_size, self.dim))
        self.p_best = self.pop.copy()
        self.p_best_val = np.full(self.population_size, np.inf)

    def differential_evolution_step(self, func):
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = self.pop[indices]
            mutant = x1 + 0.8 * (x2 - x3)
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
            if func(mutant) < func(self.pop[i]):
                self.pop[i] = mutant
                self.evaluations += 1

    def particle_swarm_optimization(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        
        fitness = np.apply_along_axis(func, 1, self.pop)
        self.evaluations += self.population_size
        self.p_best_val = np.minimum(self.p_best_val, fitness)
        better_mask = fitness < self.p_best_val
        self.p_best[better_mask] = self.pop[better_mask]
        
        if fitness.min() < self.g_best_val:
            self.g_best_val = fitness.min()
            self.g_best = self.pop[fitness.argmin()].copy()

        while self.evaluations < self.budget:
            r_p = np.random.rand(self.population_size, self.dim)
            r_g = np.random.rand(self.population_size, self.dim)

            self.velocities = (self.adaptive_inertia_weight() * self.velocities +
                               self.phi_p * r_p * (self.p_best - self.pop) +
                               self.phi_g * r_g * (self.g_best - self.pop))
            self.pop = np.clip(self.pop + self.velocities, lb, ub)

            self.differential_evolution_step(func)

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

    def local_pattern_search(self, func):
        if self.g_best is not None:
            result = minimize(func, self.g_best, method='L-BFGS-B', bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
            self.evaluations += result.nfev

            if result.fun < self.g_best_val:
                self.g_best_val = result.fun
                self.g_best = result.x

    def __call__(self, func):
        self.particle_swarm_optimization(func)
        self.local_pattern_search(func)
        return self.g_best, self.g_best_val