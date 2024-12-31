import numpy as np

class DE_Metropolis:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.population = None
        self.fitness = None
        self.scale_factor = 0.8
        self.crossover_rate = 0.7
        self.step_size = 0.5

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
    
    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                f = func(self.population[i])
                self.fitness[i] = f
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = self.population[i].copy()
                    # Adaptively adjust scale factor
                    self.scale_factor = 0.5 + 0.3 * np.random.rand()

    def differential_evolution_step(self, bounds):
        for i in range(self.population_size):
            idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
            x1, x2, x3 = self.population[idxs]
            mutant = np.clip(x1 + self.scale_factor * (x2 - x3), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            f_trial = func(trial)
            if f_trial < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = f_trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial.copy()
    
    def metropolis_hastings_step(self, func, bounds):
        for i in range(self.population_size):
            proposal = self.population[i] + self.step_size * np.random.normal(0, 1, self.dim)
            proposal = np.clip(proposal, bounds.lb, bounds.ub)
            f_proposal = func(proposal)
            if f_proposal < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - f_proposal) / abs(self.fitness[i])):
                self.population[i] = proposal
                self.fitness[i] = f_proposal
                if f_proposal < self.f_opt:
                    self.f_opt = f_proposal
                    self.x_opt = proposal.copy()
    
    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.differential_evolution_step(func.bounds)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.metropolis_hastings_step(func, func.bounds)
            evaluations += self.population_size
        return self.f_opt, self.x_opt