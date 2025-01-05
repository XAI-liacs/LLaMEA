import numpy as np

class DifferentialEvolutionACM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.initial_crossover_rate = 0.9
        self.final_crossover_rate = 0.5
        self.initial_mutation_factor = 0.8
        self.final_mutation_factor = 0.5

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            crossover_rate = (self.initial_crossover_rate - self.final_crossover_rate) * \
                             (1 - eval_count / self.budget) + self.final_crossover_rate
            mutation_factor = (self.initial_mutation_factor - self.final_mutation_factor) * \
                              (1 - eval_count / self.budget) + self.final_mutation_factor

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])

                trial_vector = np.where(np.random.rand(self.dim) < crossover_rate, mutant_vector, pop[i])
                trial_value = func(trial_vector)
                eval_count += 1

                if trial_value < fitness[i]:
                    pop[i] = trial_vector
                    fitness[i] = trial_value

                if eval_count >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return pop[best_index]