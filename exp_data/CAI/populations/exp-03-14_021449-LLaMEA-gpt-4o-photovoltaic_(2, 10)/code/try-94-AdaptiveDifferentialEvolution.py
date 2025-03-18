import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.history = []  # Store historical performance

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        initial_pop_size = 10 * self.dim
        F = 0.5
        CR = 0.9
        
        # Initialize population
        population = np.random.uniform(lb, ub, (initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.history.append(np.mean(fitness))
        
        evals = initial_pop_size
        
        # Evolutionary loop
        while evals < self.budget:
            # Change 1: Adaptive population size based on fitness diversity
            fitness_diversity = np.std(fitness) / np.mean(fitness)
            pop_size = max(5, int(initial_pop_size * (0.8 + 0.2 * fitness_diversity)))
            population = population[:pop_size]
            fitness = fitness[:pop_size]
            
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F_adapted = F * (1 + 0.15 * np.random.randn())
                mutant = np.clip(a + F_adapted * (b - c), lb, ub) 
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evals += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
            
            mean_fitness = np.mean(fitness)
            self.history.append(mean_fitness)
            if len(self.history) > 5:
                recent_perf = np.diff(self.history[-5:])
                # Change 2: Adjust F more dynamically
                F = max(0.1, min(1.0, F + 0.1 * np.sign(recent_perf[-1])))
                CR_variance = np.var(self.history[-5:])
                CR = 0.5 * (CR + min(max(0.1, CR + 0.2 * np.sign(CR_variance)), 1.0)) 
        
        best_idx = np.argmin(fitness)
        return population[best_idx]