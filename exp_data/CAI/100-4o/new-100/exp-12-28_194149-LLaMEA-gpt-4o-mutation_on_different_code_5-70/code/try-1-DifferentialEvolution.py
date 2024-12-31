import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub

        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Select three random indices different from each other and i
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Mutation
                F_adaptive = self.F * (1 - eval_count / self.budget)  # Adaptive F
                mutant = np.clip(a + F_adaptive * (b - c), lower_bound, upper_bound)

                # Crossover
                CR_adaptive = self.CR * (1 - eval_count / self.budget)  # Adaptive CR
                cross_points = np.random.rand(self.dim) < CR_adaptive
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt