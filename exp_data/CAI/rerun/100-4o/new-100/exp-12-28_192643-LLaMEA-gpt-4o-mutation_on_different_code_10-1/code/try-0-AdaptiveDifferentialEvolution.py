import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
        self.func_evals = 0

    def mutate(self, population, idx):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adapt_params(self):
        # Adaptive strategy that modifies F and CR over time
        self.F = np.abs(np.sin(self.func_evals / self.budget * np.pi))
        self.CR = 0.9 - 0.5 * (self.func_evals / self.budget)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.func_evals += self.pop_size

        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                self.adapt_params()
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                f_trial = func(trial)
                self.func_evals += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if self.func_evals >= self.budget:
                    break

        return self.f_opt, self.x_opt