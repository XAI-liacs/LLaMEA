import numpy as np

class DifferentialEvolutionAdaptivePopulation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                trial_value = func(trial)
                eval_count += 1

                if trial_value < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_value

                if eval_count >= self.budget:
                    break

            # Adapt population size based on convergence and diversity
            if eval_count < self.budget:
                fitness_std = np.std(fitness)
                if fitness_std < 1e-5:  # Convergence detected
                    self.population_size = min(self.initial_population_size, self.population_size + 1)
                else:  # Increase exploration
                    self.population_size = max(4, self.population_size - 1)

                # Adjust current population size if necessary
                if len(pop) != self.population_size:
                    pop = np.resize(pop, (self.population_size, self.dim))
                    fitness = np.resize(fitness, self.population_size)
                    if len(pop) > self.population_size:
                        excess = len(pop) - self.population_size
                        pop = np.delete(pop, np.random.choice(len(pop), excess, replace=False), axis=0)
                        fitness = np.delete(fitness, np.random.choice(len(fitness), excess, replace=False))

        best_idx = np.argmin(fitness)
        return pop[best_idx]