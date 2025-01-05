import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.f_opt:
                    self.f_opt = self.fitness[i]
                    self.x_opt = self.population[i].copy()

    def mutate(self, i):
        indices = np.arange(self.pop_size)
        indices = np.delete(indices, i)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, -5.0, 5.0)

    def crossover(self, target, mutant):
        crossover_point = np.random.randint(self.dim)
        trial = np.array([mutant[d] if np.random.rand() < self.CR or d == crossover_point else target[d] for d in range(self.dim)])
        return trial

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        self.evaluate_population(func)
        
        eval_count = self.pop_size
        while eval_count < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial.copy()
                
                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt