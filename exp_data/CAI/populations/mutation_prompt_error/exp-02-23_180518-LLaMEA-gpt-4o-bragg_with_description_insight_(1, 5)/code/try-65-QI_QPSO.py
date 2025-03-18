import numpy as np
from scipy.optimize import minimize

class QI_QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(100, 10 * dim))
        self.pop = None
        self.velocities = None
        self.p_best = None
        self.p_best_val = None
        self.g_best = None
        self.g_best_val = np.inf
        self.evaluations = 0
        self.beta = 0.5  # Quantum inspired coefficient
        self.periodicity_factor = 0.8  # Encourage periodic solutions
        self.phi_g = 0.5

    def initialize_population(self, lb, ub):
        self.pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.p_best = self.pop.copy()
        self.p_best_val = np.full(self.population_size, np.inf)

    def promote_periodicity(self, solution):
        repeat_count = self.dim // len(solution)
        periodic_solution = np.tile(solution, repeat_count)[:self.dim]
        return periodic_solution

    def quantum_inspired_swarm_optimization(self, func):
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
            r_g = np.random.rand(self.population_size, self.dim)
            
            # Quantum inspired update
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                l_best = np.mean(self.pop, axis=0)
                p = self.phi_g * r_g[i] * (self.g_best - self.pop[i]) + (1 - self.phi_g) * (l_best - self.pop[i])
                self.pop[i] = self.pop[i] + self.beta * (p - self.pop[i])
                self.pop[i] = np.clip(self.pop[i], lb, ub)

                # Apply periodic enhancement
                if np.random.rand() < self.periodicity_factor:
                    self.pop[i] = self.promote_periodicity(self.pop[i])

                fitness[i] = func(self.pop[i])
                self.evaluations += 1
                
                if fitness[i] < self.p_best_val[i]:
                    self.p_best_val[i] = fitness[i]
                    self.p_best[i] = self.pop[i]

                if fitness[i] < self.g_best_val:
                    self.g_best_val = fitness[i]
                    self.g_best = self.pop[i]

    def local_pattern_search(self, func):
        if self.g_best is not None:
            result = minimize(func, self.promote_periodicity(self.g_best), method='L-BFGS-B', bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
            self.evaluations += result.nfev

            if result.fun < self.g_best_val:
                self.g_best_val = result.fun
                self.g_best = result.x

    def __call__(self, func):
        self.quantum_inspired_swarm_optimization(func)
        self.local_pattern_search(func)
        return self.g_best, self.g_best_val