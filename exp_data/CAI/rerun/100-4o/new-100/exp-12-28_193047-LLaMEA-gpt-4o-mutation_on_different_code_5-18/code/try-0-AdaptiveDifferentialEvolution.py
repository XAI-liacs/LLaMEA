import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx]

        F = 0.5  # Initial mutation factor
        CR = 0.9  # Initial crossover rate

        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                # Mutation
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i], fitness[i] = trial, trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt, self.x_opt = trial_fitness, trial

            # Adaptive control of F and CR
            diversity = np.mean(np.std(pop, axis=0)) / np.std([lb, ub], axis=0)
            F = 0.5 + 0.5 * np.tanh(diversity - 1)
            CR = 0.9 - 0.5 * np.tanh(diversity - 1)

            if evals >= self.budget:
                break

        return self.f_opt, self.x_opt