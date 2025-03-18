import numpy as np
import scipy.optimize as opt

class ImprovedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.f = 0.8  # DE Mutation Factor
        self.cr = 0.9  # DE Crossover Rate
    
    def quasi_oppositional_init(self, lb, ub):
        """Initialize population using quasi-oppositional strategy."""
        pop = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        opp_pop = lb + ub - pop
        return np.vstack((pop, opp_pop))
    
    def periodicity_promoting_mutation(self, individual, lb, ub):
        """Promote periodicity by adding sinusoidal perturbations."""
        perturbation = np.sin(np.linspace(0, 2 * np.pi, self.dim))
        mutated = individual + 0.1 * perturbation * (ub - lb)
        return np.clip(mutated, lb, ub)
    
    def adaptive_differential_evolution(self, pop, func, lb, ub, evaluations):
        """Perform DE with adaptive population size reduction."""
        current_pop_size = max(self.initial_population_size // (1 + evaluations // self.budget), 5)
        next_pop = np.empty((current_pop_size, self.dim))
        for i in range(current_pop_size):
            indices = list(range(current_pop_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), lb, ub)
            mutant = self.periodicity_promoting_mutation(mutant, lb, ub)
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
        pop = self.quasi_oppositional_init(lb, ub)
        pop = pop[np.argsort([func(x) for x in pop])[:self.initial_population_size]]  # Select top half
        
        evaluations = self.initial_population_size * 2
        while evaluations < self.budget:
            pop = self.adaptive_differential_evolution(pop, func, lb, ub, evaluations)
            evaluations += len(pop)
            
            for i in range(len(pop)):
                if evaluations >= self.budget:
                    break
                refined = self.local_search(pop[i], func, lb, ub)
                if func(refined) < func(pop[i]):
                    pop[i] = refined
                evaluations += 1
        
        best_idx = np.argmin([func(x) for x in pop])
        return pop[best_idx]