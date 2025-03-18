import numpy as np
import scipy.optimize as opt

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f = 0.8  # DE Mutation Factor
        self.cr = 0.9  # DE Crossover Rate
        self.periodicity_weight = 0.05  # Encourage periodicity
    
    def quasi_oppositional_init(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = lb + ub - pop
        return np.vstack((pop, opp_pop))
    
    def differential_evolution(self, pop, func, lb, ub):
        next_pop = np.empty((self.population_size, self.dim))
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cr
            trial = np.where(cross_points, mutant, pop[i])
            if self.is_periodic(trial):
                trial -= self.periodicity_weight  # Periodicity encouragement
            next_pop[i] = trial if func(trial) < func(pop[i]) else pop[i]
        return next_pop

    def is_periodic(self, x):
        mid = len(x) // 2
        return np.allclose(x[:mid], x[mid:], atol=0.1)

    def local_search(self, x, func, lb, ub):
        result = opt.minimize(func, x, bounds=opt.Bounds(lb, ub), method='L-BFGS-B')
        return result.x if result.success else x
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = self.quasi_oppositional_init(lb, ub)
        pop = pop[np.argsort([func(x) for x in pop])[:self.population_size]]  

        evaluations = self.population_size * 2
        while evaluations < self.budget:
            pop = self.differential_evolution(pop, func, lb, ub)
            evaluations += self.population_size

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                refined = self.local_search(pop[i], func, lb, ub)
                if func(refined) < func(pop[i]):
                    pop[i] = refined
                evaluations += 1

        best_idx = np.argmin([func(x) for x in pop])
        return pop[best_idx]