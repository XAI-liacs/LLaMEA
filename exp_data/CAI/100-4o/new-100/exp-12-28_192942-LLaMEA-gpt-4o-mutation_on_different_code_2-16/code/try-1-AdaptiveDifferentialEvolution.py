import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self._update_best(population, fitness)

        evaluations = self.population_size
        while evaluations < self.budget:
            # Adaptive mutation factor based on diversity
            diversity = np.mean(np.std(population, axis=0))
            # Modified line to adjust mutation factor calculation
            F = np.random.uniform(0.4, 0.9) * (diversity / (ub - lb).mean())

            new_population = []
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = f_trial
                else:
                    new_population.append(population[i])

                # Update best solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                evaluations += 1
                if evaluations >= self.budget:
                    break

            population = np.array(new_population)
        
        return self.f_opt, self.x_opt

    def _update_best(self, population, fitness):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.f_opt:
            self.f_opt = fitness[min_idx]
            self.x_opt = population[min_idx]