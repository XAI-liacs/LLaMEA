import numpy as np
import scipy.optimize as opt

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # DE Mutation Factor
        self.cr = 0.9  # DE Crossover Rate
    
    def quasi_oppositional_init(self, lb, ub):
        """Initialize population using quasi-oppositional strategy."""
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = lb + ub - pop
        return np.vstack((pop, opp_pop))
    
    def differential_evolution(self, pop, func, lb, ub):
        """Perform the DE algorithm to evolve the population."""
        next_pop = np.empty((self.population_size, self.dim))
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            self.f = 0.7 + np.random.rand() * 0.3  # Adaptive Mutation Factor
            mutant = np.clip(a + self.f * (b - c), lb, ub)
            self.cr = 0.8 + np.random.rand() * 0.2  # Adaptive Crossover Rate
            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, pop[i])
            next_pop[i] = trial if self.enhanced_fitness(trial, func) < self.enhanced_fitness(pop[i], func) else pop[i]
        return next_pop
    
    def local_search(self, x, func, lb, ub):
        """Perform local search using BFGS from scipy."""
        result = opt.minimize(func, x, bounds=opt.Bounds(lb, ub), method='L-BFGS-B')
        return result.x if result.success else x
    
    def enhanced_fitness(self, x, func):
        """Enhance fitness with periodicity encouragement."""
        periodicity_penalty = np.std(x - np.roll(x, self.dim // 2))
        return func(x) + periodicity_penalty
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.quasi_oppositional_init(lb, ub)
        pop = pop[np.argsort([self.enhanced_fitness(x, func) for x in pop])[:self.population_size]]  # Select top half
        
        evaluations = self.population_size * 2
        while evaluations < self.budget:
            self.population_size = max(10, int(self.population_size * 0.9))  # Dynamic Population Adjustment
            pop = self.differential_evolution(pop, func, lb, ub)
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                refined = self.local_search(pop[i], func, lb, ub)
                if self.enhanced_fitness(refined, func) < self.enhanced_fitness(pop[i], func):
                    pop[i] = refined
                evaluations += 1
        
        best_idx = np.argmin([self.enhanced_fitness(x, func) for x in pop])
        return pop[best_idx]