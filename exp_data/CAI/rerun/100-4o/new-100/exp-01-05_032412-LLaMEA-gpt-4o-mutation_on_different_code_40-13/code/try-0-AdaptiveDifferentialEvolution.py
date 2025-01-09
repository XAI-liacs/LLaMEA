import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation with adaptive F
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Adaptive mutation factor
                F = 0.5 + np.random.rand() * 0.5
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Greedy selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    
                    # Update the best solution found
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                # Break if evaluations exceed the budget
                if evaluations >= self.budget:
                    break
                    
            # Local search around the best solution found
            if evaluations < self.budget:
                perturbed = self.x_opt + np.random.normal(0, 0.1, self.dim)
                perturbed = np.clip(perturbed, lb, ub)
                f_perturbed = func(perturbed)
                evaluations += 1
                if f_perturbed < self.f_opt:
                    self.f_opt = f_perturbed
                    self.x_opt = perturbed

        return self.f_opt, self.x_opt