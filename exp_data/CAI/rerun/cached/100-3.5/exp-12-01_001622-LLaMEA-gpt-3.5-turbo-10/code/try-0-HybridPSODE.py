import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.pso_w = 0.5
        self.pso_c1 = 1.5
        self.pso_c2 = 1.5
        self.de_f = 0.8
        self.de_cr = 0.9

    def __call__(self, func):
        def pso_update(position, velocity, pbest, gbest):
            new_velocity = self.pso_w * velocity + self.pso_c1 * np.random.rand(self.dim) * (pbest - position) + self.pso_c2 * np.random.rand(self.dim) * (gbest - position)
            new_position = position + new_velocity
            return new_position, new_velocity

        def de_mutation(population, target_idx):
            candidates = [idx for idx in range(self.pop_size) if idx != target_idx]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            mutant = np.clip(a + self.de_f * (b - c), -5.0, 5.0)
            return mutant

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(individual) for individual in population]
        best_idx = np.argmin(fitness)
        gbest = population[best_idx].copy()
        pbest = population.copy()
        pbest_fitness = fitness.copy()

        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                # PSO update
                population[i], velocity = pso_update(population[i], velocity, pbest[i], gbest)

                # DE mutation
                mutant = de_mutation(population, i)

                # DE crossover
                crossover_mask = np.random.rand(self.dim) < self.de_cr
                trial = np.where(crossover_mask, mutant, population[i])

                # Update if better
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < pbest_fitness[i]:
                        pbest[i] = trial
                        pbest_fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        gbest = trial
                        best_idx = i

        return gbest