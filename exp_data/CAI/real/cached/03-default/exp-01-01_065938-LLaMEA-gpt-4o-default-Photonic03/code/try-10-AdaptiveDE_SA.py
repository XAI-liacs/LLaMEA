import numpy as np

class AdaptiveDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.temp_max = 1.0
        self.temp_min = 0.1
        self.temp = self.temp_max
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_global = pop[np.argmin(fitness)]
        best_global_fitness = np.min(fitness)

        evaluations = self.population_size

        while evaluations < self.budget:
            temp_schedule = (self.temp_max - self.temp_min) * (1 - evaluations / self.budget) + self.temp_min
            new_pop = np.zeros_like(pop)

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + self.mutation_factor * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, pop[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temp_schedule):
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_pop[i] = pop[i]

            pop = new_pop
            evaluations += self.population_size

            if np.min(fitness) < best_global_fitness:
                best_global = pop[np.argmin(fitness)]
                best_global_fitness = np.min(fitness)

            self.history.append(best_global)

        return best_global