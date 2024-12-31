import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.F = 0.5  # Mutation factor
        self.CR = 0.7  # Crossover rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            new_population = []
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, bounds[0], bounds[1])
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):  # Ensure at least one dimension is crossed over
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population.append(population[i])

                if evals >= self.budget:
                    break

            population = np.array(new_population)
            self.adapt_strategy(fitness)
        
        return self.f_opt, self.x_opt

    def adapt_strategy(self, fitness):
        sorted_indices = np.argsort(fitness)
        best_indices = sorted_indices[:int(0.2 * self.population_size)]
        if len(best_indices) > 0:
            self.F = np.mean(np.random.uniform(0.4, 0.9, len(best_indices)))
            self.CR = np.mean(np.random.uniform(0.5, 0.9, len(best_indices)))