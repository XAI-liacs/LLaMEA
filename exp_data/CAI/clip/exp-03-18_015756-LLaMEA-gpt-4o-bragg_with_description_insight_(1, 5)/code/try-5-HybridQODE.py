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
        pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        opp_pop = self.lb + self.ub - pop
        return np.vstack((pop, opp_pop))

    def adaptive_mutation(self, current_fitness, best_fitness):
        return 0.5 + 0.3 * np.exp(-(best_fitness - current_fitness))

    def enhanced_local_search(self, solution, func):
        result = minimize(func, solution, bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)], method='SLSQP')
        return result.x if result.success else solution

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        evals = 0
        pop = self.quasi_oppositional_initialization()
        best_solution = None
        best_fitness = float('inf')

        while evals < self.budget:
            new_pop = np.zeros_like(pop)
            for i in range(len(pop)):
                indices = np.random.choice(len(pop), 3, replace=False)
                x_1, x_2, x_3 = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                current_fitness = func(pop[i])
                self.f = self.adaptive_mutation(current_fitness, best_fitness)
                mutant = np.clip(x_1 + self.f * (x_2 - x_3), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.cr
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
                    refined_solution = self.enhanced_local_search(pop[i], func)
                    refined_fitness = func(refined_solution)
                    evals += 1
                    if refined_fitness < best_fitness:
                        best_fitness = refined_fitness
                        best_solution = refined_solution

        return best_solution