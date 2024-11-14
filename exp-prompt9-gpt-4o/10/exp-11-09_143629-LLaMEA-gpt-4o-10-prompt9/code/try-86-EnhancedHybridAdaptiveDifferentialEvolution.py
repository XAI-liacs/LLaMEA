import numpy as np

class EnhancedHybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = pop_size if pop_size is not None else 8 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.eval_count = 0
        self.global_best = None
        self.scaling_factor = 0.5 + np.random.rand() * 0.6  # Adaptive scaling factor
        self.dynamic_pop_size = self.pop_size

    def evaluate_population(self, func):
        for i in range(self.dynamic_pop_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1
                if self.global_best is None or self.fitness[i] < self.fitness[self.global_best]:
                    self.global_best = i

    def select_parents(self):
        indices = np.random.choice(self.dynamic_pop_size, 3, replace=False)
        return self.population[indices]

    def mutate(self, a, b, c):
        mutant = np.clip(a + self.scaling_factor * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, CR=0.8):  # Adjusted crossover rate
        crossover_mask = np.random.rand(self.dim) < CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def guided_search(self):
        direction = np.random.uniform(-1.0, 1.0, self.dim)
        guide = np.clip(self.population[self.global_best] + 0.15 * direction, self.lower_bound, self.upper_bound)
        return guide

    def competitive_selection(self, trial, trial_fitness, target_idx):
        if trial_fitness < self.fitness[target_idx]:
            return trial, trial_fitness
        else:
            return self.population[target_idx], self.fitness[target_idx]

    def optimize(self, func):
        self.evaluate_population(func)
        while self.eval_count < self.budget:
            for i in range(self.dynamic_pop_size):
                if np.random.rand() < 0.3:  # Increased guided search probability
                    trial = self.guided_search()
                else:
                    target = self.population[i]
                    a, b, c = self.select_parents()
                    mutant = self.mutate(a, b, c)
                    trial = self.crossover(target, mutant)

                trial_fitness = func(trial)
                self.eval_count += 1

                self.population[i], self.fitness[i] = self.competitive_selection(trial, trial_fitness, i)

                if self.fitness[i] < self.fitness[self.global_best]:
                    self.global_best = i

                if self.eval_count >= self.budget:
                    break

            self.dynamic_pop_size = max(int(self.pop_size * (1.0 - 0.1 * (self.eval_count / self.budget))), 4)  # Dynamic population reduction

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def __call__(self, func):
        best_solution, best_fitness = self.optimize(func)
        return best_solution, best_fitness