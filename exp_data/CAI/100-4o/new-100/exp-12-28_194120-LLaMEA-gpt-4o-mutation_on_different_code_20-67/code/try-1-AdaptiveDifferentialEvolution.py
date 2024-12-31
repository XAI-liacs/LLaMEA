import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()

        func_evals = self.pop_size
        mutation_scale = np.random.rand() * 0.5 + 0.5  # Self-adaptive mutation
        crossover_rate = np.random.rand() * 0.2 + 0.7  # Self-adaptive crossover

        while func_evals < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                while i in idxs:
                    idxs = np.random.choice(self.pop_size, 3, replace=False)

                a, b, c = population[idxs]
                mutant = np.clip(a + mutation_scale * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                func_evals += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial.copy()

                # Adapt mutation_scale and crossover_rate based on diversity
                if func_evals < self.budget:
                    diversity = np.std(population, axis=0).mean()
                    mutation_scale = np.clip(mutation_scale + np.random.normal(0, 0.1), 0.4, 1.0)
                    crossover_rate = np.clip(crossover_rate + np.random.normal(0, 0.05), 0.6, 1.0)

        return self.f_opt, self.x_opt