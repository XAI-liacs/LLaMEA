import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, population, target_idx):
        candidates = population[[idx for idx in range(len(population)) if idx != target_idx]]
        a, b, c = np.random.choice(len(candidates), 3, replace=False)
        mutant_vector = population[a] + self.f * (population[b] - population[c])
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.cr
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def select(self, func, trial_vector, target_vector):
        target_fitness = func(target_vector)
        trial_fitness = func(trial_vector)

        if trial_fitness < target_fitness:
            return trial_vector, trial_fitness
        else:
            return target_vector, target_fitness

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        for _ in range(self.budget):
            new_population = np.empty_like(population)

            for i, target_vector in enumerate(population):
                mutant_vector = self.mutate(population, i)
                trial_vector = self.crossover(target_vector, mutant_vector)
                selected_vector, selected_fitness = self.select(func, trial_vector, target_vector)

                new_population[i] = selected_vector

                if selected_fitness < self.f_opt:
                    self.f_opt = selected_fitness
                    self.x_opt = selected_vector

            population = new_population

        return self.f_opt, self.x_opt