import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def quasi_oppositional_de(self, func, bounds, pop_size=20, F=0.45, CR=0.98, max_iter=None):  
        if max_iter is None:
            max_iter = self.budget // pop_size

        def periodic_penalty(x, gen, max_iter):
            penalty = 0.0
            dynamic_factor = 1 + 0.7 * (gen / max_iter)  # Adjusted penalty factor
            for i in range(1, len(x)):
                diff = abs(x[i] - x[i-1]) % (bounds.ub[0] - bounds.lb[0])
                penalty += (diff - 0.2) ** 2 * dynamic_factor  # Adjust penalty with dynamic factor
            return penalty

        def quasi_opposite(x, lb, ub):
            return lb + ub - x
        
        population = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        quasi_population = quasi_opposite(population, bounds.lb, bounds.ub)
        population = np.concatenate((population, quasi_population), axis=0)
        
        scores = np.empty(2 * pop_size)
        for i in range(2 * pop_size):
            scores[i] = func(population[i]) + periodic_penalty(population[i], 0, max_iter)
            self.eval_count += 1

        for gen in range(max_iter):  # Change: Added iteration progress
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break

                indices = [idx for idx in range(2 * pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = 0.3 + (0.4 * (gen / max_iter)) * np.random.rand() * (1 - gen/max_iter)  # Modified mutation factor
                mutant = np.clip(a + F_dynamic * (b - c), bounds.lb, bounds.ub)
                
                CR_dynamic = CR * (1 - (np.std(population.ravel()) / (np.mean(population.ravel()) + 1e-9)))  # Modified crossover probability
                cross_points = np.random.rand(self.dim) < CR_dynamic
                
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                trial_score = func(trial) + periodic_penalty(trial, gen, max_iter)  # Updated penalty call
                self.eval_count += 1

                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

            if gen % 5 == 0:  # Local search every 5 generations
                loc_idx = np.argmin(scores[:pop_size])
                local_res = minimize(func, population[loc_idx], method='L-BFGS-B',  # Added local search
                                     bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                                     options={'maxiter': 5})
                if local_res.fun + periodic_penalty(local_res.x, gen, max_iter) < scores[loc_idx]:
                    population[loc_idx] = local_res.x
                    scores[loc_idx] = local_res.fun

        best_idx = np.argmin(scores[:pop_size])
        return population[best_idx]

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.quasi_oppositional_de(func, bounds)
        
        if self.eval_count < self.budget:
            res = minimize(func, best_solution, method='Nelder-Mead', bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                           options={'maxfev': self.budget - self.eval_count})
            best_solution = res.x
            self.eval_count += res.nfev

        return best_solution