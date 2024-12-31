import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.cr = 0.9  # Crossover probability
        self.initial_f = 0.5  # Initial mutation factor
        self.bounds = np.array([-5.0, 5.0])

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while self.budget > 0:
            f = self.adaptive_mutation_factor(fitness)
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + f * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])
                
                # Opposition-based learning
                opposite = self.bounds[0] + (self.bounds[1] - trial)
                opposite = np.clip(opposite, self.bounds[0], self.bounds[1])
                f_opposite = func(opposite)
                self.budget -= 1

                if f_opposite < fitness[i]:
                    trial = opposite
                    f_trial = f_opposite
                else:
                    f_trial = func(trial)
                    self.budget -= 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population[i] = population[i]

                if self.budget <= 0:
                    break

            population = new_population

        return self.f_opt, self.x_opt

    def adaptive_mutation_factor(self, fitness):
        # Adaptively adjust the mutation factor based on fitness performance
        improvement = np.diff(np.sort(fitness))
        f_dynamic = np.clip(self.initial_f * (1 + improvement.mean()), 0.1, 0.9)
        return f_dynamic