import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.bounds = (-100.0, 100.0)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        success_history = []
        mutation_rate = 0.5
        crossover_rate = 0.9

        while evaluations < self.budget:
            diversity = np.mean(np.std(population, axis=0))
            success_rate = np.mean(success_history[-10:]) if success_history else 0.5
            
            # Self-adaptive mutation and crossover rates
            mutation_rate = 0.4 + 0.6 * success_rate
            crossover_rate = 0.8 + 0.2 * success_rate

            # Dynamic population size adjustment
            if evaluations % (self.population_size * 10) == 0:
                if diversity < 0.1 and success_rate < 0.3:
                    self.population_size = max(20, int(self.population_size * 0.9))
                elif success_rate > 0.7:
                    self.population_size = min(100, int(self.population_size * 1.1))

            new_population = np.empty_like(population)

            for i in range(self.population_size):
                idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = population[idxs]

                # Mutation
                fit_diff = np.abs(fitness[idxs[0]] - fitness[idxs[1]])
                adaptive_mutation_rate = mutation_rate * (1 + 0.5 * (fit_diff / (1 + np.std(fitness))))
                mutant = x1 + adaptive_mutation_rate * (x2 - x3)

                # Boundary Handling
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    success_history.append(1)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population[i] = population[i]
                    success_history.append(0)

                evaluations += 1
                if evaluations >= self.budget:
                    break

            population = new_population

        return self.f_opt, self.x_opt