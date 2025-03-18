import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.history = []  # Store historical performance

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        pop_size = 10 * self.dim  # A common heuristic choice
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability
        
        # Initialize population
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.history.append(np.mean(fitness))
        
        evals = pop_size
        
        # Evolutionary loop
        while evals < self.budget:
            for i in range(pop_size):
                # Mutation and Crossover
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.6 * F * (b - c), lb, ub)  # Modified line
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evals += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
            
            # Adaptation based on history
            mean_fitness = np.mean(fitness)
            self.history.append(mean_fitness)
            if len(self.history) > 5:
                recent_perf = np.diff(self.history[-5:])
                F = min(max(0.1, F + 0.1 * np.sign(recent_perf[-1])), 1.0)
                CR = min(max(0.1, CR + 0.05 * np.sign(recent_perf[-1])), 1.0)  # Modified line
        
        best_idx = np.argmin(fitness)
        return population[best_idx]