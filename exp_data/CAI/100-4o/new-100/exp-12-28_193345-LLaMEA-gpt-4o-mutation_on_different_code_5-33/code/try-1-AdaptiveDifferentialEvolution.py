import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.adaptive_step = 0.05

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation: select three random indices different from i
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])

                # Ensure mutant is within bounds
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                # Evaluate the trial vector
                f_trial = func(trial)
                eval_count += 1

                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adaptive adjustment of parameters
                if eval_count % (self.budget // 20) == 0:  # Changed from self.budget // 10 to self.budget // 20
                    self.mutation_factor = max(0.1, self.mutation_factor + np.random.uniform(-self.adaptive_step, self.adaptive_step))
                    self.crossover_rate = np.clip(self.crossover_rate + np.random.uniform(-self.adaptive_step, self.adaptive_step), 0.1, 1.0)

        return self.f_opt, self.x_opt