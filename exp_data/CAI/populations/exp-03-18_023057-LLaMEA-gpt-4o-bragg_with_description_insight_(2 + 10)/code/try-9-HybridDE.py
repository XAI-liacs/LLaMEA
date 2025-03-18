import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.eval_count = 0
    
    def quasi_oppositional_init(self, lb, ub):
        mid = (lb + ub) / 2
        range_ = ub - lb
        return np.random.uniform(mid - range_/2, mid + range_/2, (self.pop_size, self.dim))
    
    def evaluate(self, pop, func):
        scores = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            scores[i] = func(pop[i])
            self.eval_count += 1
        return scores
    
    def differential_evolution(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = self.quasi_oppositional_init(lb, ub)
        scores = self.evaluate(pop, func)
        
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Mutation
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                # Crossover
                self.CR = 0.5 + 0.3 * np.random.randn()  # Adaptive CR
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                score_trial = func(trial)
                self.eval_count += 1
                if score_trial < scores[i]:
                    pop[i] = trial
                    scores[i] = score_trial
        
        return pop, scores

    def local_search(self, candidate, func, bounds):
        res = minimize(func, candidate, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        return res.x, res.fun

    def __call__(self, func):
        bounds = func.bounds
        pop, scores = self.differential_evolution(func, bounds)
        
        # Local search on best candidates
        best_idx = np.argmin(scores)
        best_candidate = pop[best_idx]
        best_score = scores[best_idx]
        
        best_candidate, best_score = self.local_search(best_candidate, func, bounds)
        
        return best_candidate