import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def de_opponent(pop, F, CR):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            mutant = np.clip(a + F * (b - c), -5.0, 5.0)
            cross_points = np.random.rand(self.dim) < CR
            trial = np.where(cross_points, mutant, pop)
            return trial

        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        F, CR = 0.5, 0.9
        T0, alpha = 1.0, 0.99

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(pop_size):
                trial = de_opponent(pop, F, CR)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i], fitness[i] = trial, trial_fitness

            T = T0 * alpha**_
            for i in range(pop_size):
                current, proposed = pop[i], np.clip(pop[i] + np.random.normal(0, T, self.dim), -5.0, 5.0)
                current_fitness, proposed_fitness = fitness[i], func(proposed)
                if proposed_fitness < current_fitness:
                    pop[i], fitness[i] = proposed, proposed_fitness

        best_idx = np.argmin(fitness)
        return pop[best_idx]