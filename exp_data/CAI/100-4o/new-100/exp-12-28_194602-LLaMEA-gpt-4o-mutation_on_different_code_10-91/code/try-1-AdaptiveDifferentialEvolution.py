import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                # Mutation: Randomly select three distinct individuals
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[idxs]
                # Self-adaptive mutation factor
                self.F = np.random.normal(0.5, 0.3)
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update the best solution found
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                # Adapt F and CR based on success rate
                success_rate = np.sum(fitness < trial_fitness) / (evals + 1)
                self.CR = np.clip(self.CR + 0.1 * (0.9 - success_rate), 0.1, 1.0)
                
                # Elitism: Preserve the current best solution
                if fitness[i] > self.f_opt:
                    population[i] = self.x_opt

        return self.f_opt, self.x_opt