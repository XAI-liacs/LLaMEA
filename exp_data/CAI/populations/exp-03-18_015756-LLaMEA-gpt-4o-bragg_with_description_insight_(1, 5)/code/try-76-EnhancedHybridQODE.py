import numpy as np
from scipy.optimize import minimize

class EnhancedHybridQODE:
    def __init__(self, budget, dim, pop_size=50, f=0.8, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.lb = None
        self.ub = None
        self.history = []

    def quasi_oppositional_initialization(self):
        pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        opp_pop = self.lb + self.ub - pop
        return np.vstack((pop, opp_pop))

    def harmonic_mutation(self, solution, evals):
        freq = 1.0 + np.sin(2 * np.pi * evals / self.budget) * 0.5
        return solution + 0.01 * np.random.randn(self.dim) * freq

    def cultural_influence(self, pop):
        if self.history:
            best_hist = min(self.history, key=lambda x: x[1])[0]
            return pop + 0.05 * (best_hist - pop)
        return pop

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        evals = 0
        pop = self.quasi_oppositional_initialization()
        best_solution = None
        best_fitness = float('inf')
        
        while evals < self.budget:
            self.pop_size = max(5, int(self.pop_size * (1 - evals / (1.5 * self.budget))))
            new_pop = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                if np.random.rand() < (0.3 + 0.7 * evals / self.budget):
                    mutant = self.harmonic_mutation(pop[i], evals)
                else:
                    mutant = self.levy_flight(pop[i])
                
                cr_dynamic = self.cr * (1 - evals / self.budget)
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
            
            new_pop = self.cultural_influence(new_pop)
            pop = new_pop
            self.history.append((best_solution, best_fitness))

            for i in range(len(pop)):
                if evals < self.budget:
                    refined_solution = self.local_search(pop[i], func)
                    refined_fitness = func(refined_solution)
                    evals += 1
                    if refined_fitness < best_fitness:
                        best_fitness = refined_fitness
                        best_solution = refined_solution

        return best_solution