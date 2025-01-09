import numpy as np

class ADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.8  # Scaling factor for mutation
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

            # Local search using a simple exploitation strategy
            best_idx = np.argmin(fitness)
            local_best = population[best_idx]
            for _ in range(self.dim):
                step = np.random.uniform(-0.1, 0.1, self.dim)
                local_candidate = np.clip(local_best + step, self.lower_bound, self.upper_bound)
                local_candidate_fitness = func(local_candidate)
                eval_count += 1

                if local_candidate_fitness < fitness[best_idx]:
                    local_best = local_candidate
                    fitness[best_idx] = local_candidate_fitness

                if eval_count >= self.budget:
                    break

            population[best_idx] = local_best

        best_fitness = np.min(fitness)
        best_solution = population[np.argmin(fitness)]
        return best_solution, best_fitness