import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size
        self.f_opt, self.x_opt = np.min(fitness), population[np.argmin(fitness)]

        while eval_count < self.budget:
            F = np.random.normal(0.5, 0.3)  # adaptive control of mutation factor
            CR = np.random.uniform(0.5, 1.0)  # adaptive control of crossover probability
            new_population = np.copy(population)

            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial

                if f_trial < self.f_opt:
                    self.f_opt, self.x_opt = f_trial, trial

                if eval_count >= self.budget:
                    break

            population = new_population
            if np.std(fitness) < 1e-5:  # Adjust population size based on diversity
                self.pop_size = max(20, self.pop_size - 1)

        return self.f_opt, self.x_opt