import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evals = self.pop_size
        success_count = 0  # Track successful trials
        trials = 0  # Track total trials

        while evals < self.budget:
            best_idx = np.argmin(fitness)  # Elitism: track the best solution
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                f_trial = func(trial)
                evals += 1
                trials += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_count += 1  # Increase success count
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            population[best_idx] = self.x_opt  # Ensure elitism

            if evals % (self.pop_size * 10) == 0:
                success_rate = success_count / trials
                if success_rate > 0.2:
                    self.F = min(self.F + 0.1, 1.0)
                    self.CR = max(self.CR - 0.1, 0.1)
                else:
                    self.F = max(self.F - 0.1, 0.1)
                    self.CR = min(self.CR + 0.1, 1.0)
                success_count = 0
                trials = 0

        return self.f_opt, self.x_opt