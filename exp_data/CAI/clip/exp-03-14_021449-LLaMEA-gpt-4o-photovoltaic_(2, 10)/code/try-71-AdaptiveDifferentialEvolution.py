import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.history = []  # Store historical performance

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        initial_pop_size = 10 * self.dim  # A common heuristic choice
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability
        
        # Initialize population
        population = np.random.uniform(lb, ub, (initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.history.append(np.mean(fitness))
        
        evals = initial_pop_size
        best_idx = np.argmin(fitness)  # Track best individual
        best_solution = population[best_idx].copy()  # Retain best solution
        
        # Evolutionary loop
        while evals < self.budget:
            if len(self.history) > 1:
                pop_size = max(5, int(initial_pop_size * (0.9 + 0.1 * np.sign(np.diff(self.history[-2:])[0]))))  # Dynamic adaptation
            else:
                pop_size = initial_pop_size
            population = population[:pop_size]
            fitness = fitness[:pop_size]
            for i in range(pop_size):
                # Mutation and Crossover
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.9 * F * (b - c), lb, ub)  # Adjusted mutation factor
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
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best_solution = population[best_idx].copy()  # Update best solution
            
            # Adaptation based on history
            mean_fitness = np.mean(fitness)
            self.history.append(mean_fitness)
            if len(self.history) > 5:
                recent_perf = np.diff(self.history[-5:])
                F = 0.5 * (F + min(max(0.1, F + 0.1 * np.sign(recent_perf[-1])), 1.0))
                CR = np.clip(np.std(population, axis=0).mean(), 0.1, 1.0)  # Adaptive CR based on diversity
        
        return best_solution