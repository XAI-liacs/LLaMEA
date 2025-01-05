import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x_1, x_2, x_3 = population[indices]
                mutant = np.clip(x_1 + self.F * (x_2 - x_3), lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt, self.x_opt = f_trial, trial

                if evaluations >= self.budget:
                    break
                
                # Adaptive Random Walks
                if np.random.rand() < 0.1:
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    adaptive_solution = np.clip(self.x_opt + perturbation, lb, ub)
                    f_adaptive = func(adaptive_solution)
                    evaluations += 1
                    if f_adaptive < self.f_opt:
                        self.f_opt, self.x_opt = f_adaptive, adaptive_solution

        return self.f_opt, self.x_opt