import numpy as np
import scipy.optimize as opt

class CooperativeCoevolutionaryMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # DE Mutation Factor
        self.cr = 0.9  # DE Crossover Rate
        self.periodicity = 5  # Inject periodic solutions every 'periodicity' generations
    
    def periodic_solution_injection(self, lb, ub):
        """Generate periodic Bragg mirror solutions."""
        half_period = self.dim // 2
        period = np.concatenate((np.full(half_period, lb), np.full(half_period, ub)))
        if self.dim % 2 != 0:
            period = np.append(period, np.random.uniform(lb, ub))
        return np.tile(period, self.dim // len(period) + 1)[:self.dim]
    
    def cooperative_evolution(self, pop, func, lb, ub):
        """Perform adaptive differential evolution with cooperative strategy."""
        next_pop = np.empty((self.population_size, self.dim))
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            adaptive_f = self.f * np.random.rand()  # Adaptive mutation factor
            mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, pop[i])
            next_pop[i] = trial if func(trial) < func(pop[i]) else pop[i]
        return next_pop
    
    def local_search(self, x, func, lb, ub):
        """Perform local search using BFGS from scipy."""
        result = opt.minimize(func, x, bounds=opt.Bounds(lb, ub), method='L-BFGS-B')
        return result.x if result.success else x
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        evaluations = self.population_size
        
        while evaluations < self.budget:
            pop = self.cooperative_evolution(pop, func, lb, ub)
            evaluations += self.population_size
            
            if evaluations // self.population_size % self.periodicity == 0:
                periodic_sol = self.periodic_solution_injection(lb, ub)
                if func(periodic_sol) < max(func(x) for x in pop):
                    worst_idx = np.argmax([func(x) for x in pop])
                    pop[worst_idx] = periodic_sol
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                refined = self.local_search(pop[i], func, lb, ub)
                if func(refined) < func(pop[i]):
                    pop[i] = refined
                evaluations += 1
        
        best_idx = np.argmin([func(x) for x in pop])
        return pop[best_idx]