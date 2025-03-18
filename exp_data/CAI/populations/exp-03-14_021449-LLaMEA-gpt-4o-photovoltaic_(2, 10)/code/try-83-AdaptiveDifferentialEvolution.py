import numpy as np
import scipy.stats

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
        
        population = np.random.uniform(lb, ub, (initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.history.append(np.mean(fitness))
        
        evals = initial_pop_size
        no_improvement = 0
        
        # Chaotic mapping initialization
        chaotic_seq = np.random.rand(self.dim)
        
        while evals < self.budget:
            pop_size = max(5, int(initial_pop_size * (0.9 + 0.1 * np.sign(np.diff(self.history[-2:]).sum()))))
            population = population[:pop_size]
            fitness = fitness[:pop_size]
            F_adapt = 0.1 + chaotic_seq.mean() * 0.4  # Self-adaptive F (Change 1)
            
            for i in range(pop_size):
                levy_step = scipy.stats.levy.rvs(size=self.dim)
                a, b, c = population[np.random.choice(pop_size, 3, replace=False)]
                chaotic_seq = 4 * chaotic_seq * (1 - chaotic_seq)  # Logistic map (Change 2)

                # Chaotic mutation step (Change 3)
                chaotic_factor = 1.0 + 0.5 * (chaotic_seq - 0.5)
                mutant = np.clip(a + F_adapt * chaotic_factor * levy_step * (b - c), lb, ub)
                
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evals += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    no_improvement = 0
                else:
                    no_improvement += 1
            
            mean_fitness = np.mean(fitness)
            self.history.append(mean_fitness)
            if len(self.history) > 5:
                recent_perf = np.diff(self.history[-5:])
                F = 0.5 * (F + min(max(0.1, F + 0.1 * np.sign(recent_perf[-1])), 1.0))
                CR_variance = np.var(self.history[-5:])
                CR = 0.5 * (CR + min(max(0.1, CR + 0.1 * np.sign(CR_variance)), 1.0))
            
            if no_improvement > pop_size // 2:
                population = np.random.uniform(lb, ub, (initial_pop_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                no_improvement = 0
        
        best_idx = np.argmin(fitness)
        return population[best_idx]