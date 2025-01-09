import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, pop[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                eval_count += 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                # Update mutation factor adaptively
                self.mutation_factor = 0.5 + 0.5 * (1 - eval_count / self.budget)

                if eval_count >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return pop[best_index], fitness[best_index]