import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = 0.5  # Initial scale factor
        self.CR = 0.9  # Initial crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds_lb, bounds_ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(bounds_lb, bounds_ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation: DE/rand/1
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, bounds_lb, bounds_ub)

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

                    # Update global best
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Adaptation of F and CR
            self.F = np.clip(np.random.normal(self.F, 0.05), 0.1, 0.9)  # Narrowed standard deviation
            self.CR = np.clip(np.random.normal(self.CR, 0.05), 0.05, 0.9) # Lowered min CR for diversity

        return self.f_opt, self.x_opt