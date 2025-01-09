import numpy as np

class DELS:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 20
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        pop_fitness = np.array([func(ind) for ind in pop])
        
        eval_count = self.pop_size
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                # Adaptive Differential Evolution: Mutation and Crossover
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = pop[idxs]
                self.F = 0.5 + 0.1 * np.sin(eval_count / self.budget * np.pi)  # Adaptive F
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Refined Local Search
                local_step = (np.random.normal(size=self.dim) * 0.05)
                trial = np.clip(trial + local_step, bounds[0], bounds[1])
                
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt