import numpy as np
from scipy.optimize import minimize

class HybridQODE:
    def __init__(self, budget, dim, pop_size=50, f=0.8, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f  # Differential weight
        self.cr = cr  # Crossover probability
        self.lb = None
        self.ub = None

    def quasi_oppositional_initialization(self):
        # Initialize population with quasi-oppositional solutions
        pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        opp_pop = self.lb + self.ub - pop
        return np.vstack((pop, opp_pop))

    def local_search(self, solution, func):
        # Local search using BFGS to fine-tune solutions
        result = minimize(func, solution, bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x if result.success else solution

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        evals = 0
        pop = self.quasi_oppositional_initialization()
        best_solution = None
        best_fitness = float('inf')

        while evals < self.budget:
            self.pop_size = max(5, int(self.pop_size * (1 - evals / (1.5 * self.budget))))  # Modified line
            new_pop = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):  # Adjusting loop according to new pop_size
                indices = np.random.choice(len(pop), 3, replace=False)
                x_1, x_2, x_3 = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                f_dynamic = self.f * (0.5 + 0.5 * np.sin(np.pi * evals / self.budget))  # Modified line
                chaotic_factor = 0.6 + 0.4 * np.sin(np.pi * evals / self.budget) * np.cos(np.pi * evals / self.budget)  # Modified line
                periodic_factor = np.sin(2 * np.pi * np.sum(x_1) / self.dim)  # New line
                mutant = np.clip(x_1 + f_dynamic * chaotic_factor * periodic_factor * (x_2 - x_3), self.lb, self.ub)  # Modified line
                cr_dynamic = self.cr * (1 - evals / self.budget)  # Changed line: Adaptive crossover probability
                cross_points = np.random.rand(self.dim) < cr_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Evaluate and select
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < func(pop[i]):
                    new_pop[i] = trial
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                else:
                    new_pop[i] = pop[i]
            
            pop = new_pop

            # Periodic encouragement via local search
            for i in range(len(pop)):
                if evals < self.budget:
                    refined_solution = self.local_search(pop[i], func)
                    refined_fitness = func(refined_solution)
                    evals += 1
                    if refined_fitness < best_fitness:
                        best_fitness = refined_fitness
                        best_solution = refined_solution

        return best_solution