import numpy as np
from scipy.optimize import minimize

class HybridQODE:
    def __init__(self, budget, dim, pop_size=50, f=0.8, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.lb = None
        self.ub = None

    def quasi_oppositional_initialization(self):
        pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        opp_pop = self.lb + self.ub - pop
        return np.vstack((pop, opp_pop))

    def levy_flight(self, solution):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return solution + 0.01 * step

    def local_search(self, solution, func):
        result = minimize(func, solution, bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        evals = 0
        pop = self.quasi_oppositional_initialization()
        best_solution = None
        best_fitness = float('inf')

        while evals < self.budget:
            self.pop_size = max(5, int(self.pop_size * (1 - evals / (1.8 * self.budget))))  # Changed line
            new_pop = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                if np.random.rand() < (0.3 + 0.7 * evals / self.budget):  # Dynamic strategy selection
                    indices = np.random.choice(len(pop), 3, replace=False)
                    x_1, x_2, x_3 = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                    f_dynamic = self.f * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget)) * (1 - best_fitness / 100)
                    chaotic_factor = 0.7 + 0.3 * np.cos(np.pi * evals / self.budget) * np.sin(np.pi * evals / self.budget)
                    mutant = np.clip(x_1 + f_dynamic * chaotic_factor * (x_2 - x_3), self.lb, self.ub)
                else:
                    mutant = self.levy_flight(pop[i])
                
                cr_dynamic = self.cr * (1 - np.sin(np.pi * evals / self.budget))  # Adjusted dynamic crossover rate
                cross_points = np.random.rand(self.dim) < cr_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
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

            for i in range(len(pop)):
                if evals < self.budget:
                    refined_solution = self.local_search(pop[i], func)
                    refined_fitness = func(refined_solution)
                    evals += 1
                    if refined_fitness < best_fitness:
                        best_fitness = refined_fitness
                        best_solution = refined_solution

        return best_solution