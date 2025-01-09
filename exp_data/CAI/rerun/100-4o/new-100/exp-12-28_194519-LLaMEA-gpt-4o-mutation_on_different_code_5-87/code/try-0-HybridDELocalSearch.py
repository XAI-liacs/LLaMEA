import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9 # Crossover probability
        self.local_search_prob = 0.2

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        pop = np.random.uniform(low=bounds[0], high=bounds[1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(self.dim)] = True
                trial = np.where(crossover, mutant, pop[i])

                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Local Search
                if np.random.rand() < self.local_search_prob:
                    local_trial = np.clip(trial + np.random.normal(0, 0.1, self.dim), bounds[0], bounds[1])
                    f_local_trial = func(local_trial)
                    eval_count += 1
                    if f_local_trial < fitness[i]:
                        pop[i] = local_trial
                        fitness[i] = f_local_trial
                        if f_local_trial < self.f_opt:
                            self.f_opt = f_local_trial
                            self.x_opt = local_trial

        return self.f_opt, self.x_opt