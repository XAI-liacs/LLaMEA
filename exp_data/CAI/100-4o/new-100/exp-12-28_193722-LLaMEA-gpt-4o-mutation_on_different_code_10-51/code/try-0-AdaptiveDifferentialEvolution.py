import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        self.f_opt, self.x_opt = fitness.min(), population[fitness.argmin()]
        
        count_evals = self.pop_size
        F = 0.5
        CR = 0.9
        
        while count_evals < self.budget:
            for i in range(self.pop_size):
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                count_evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt, self.x_opt = trial_fitness, trial

                if count_evals >= self.budget:
                    break

            # Adapt F and CR based on success of trials
            successful_trials = fitness < np.array([func(ind) for ind in population])
            if np.any(successful_trials):
                F = np.clip(np.mean(successful_trials) + 0.1, 0.1, 1.9)
                CR = np.clip(np.mean(successful_trials) + 0.1, 0.1, 1.0)
        
        return self.f_opt, self.x_opt