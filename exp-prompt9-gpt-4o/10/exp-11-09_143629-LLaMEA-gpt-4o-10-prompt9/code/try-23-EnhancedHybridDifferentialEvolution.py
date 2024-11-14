import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = pop_size if pop_size is not None else 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.eval_count = 0
        self.global_best = None
        self.local_best = np.copy(self.population)
        self.scaling_factor = np.full(self.pop_size, 0.5 + np.random.rand() * 0.5)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1
                if self.global_best is None or self.fitness[i] < self.fitness[self.global_best]:
                    self.global_best = i

    def select_parents(self):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        return self.population[indices]

    def mutate(self, a, b, c, F):
        perturbation = np.random.normal(0, 0.1, self.dim)  # Injecting Gaussian noise
        mutant = np.clip(a + F * (b - c) + perturbation, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, CR=0.9):
        crossover_mask = np.random.rand(self.dim) < CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def adaptive_random_search(self):
        perturbation = np.random.uniform(-0.5, 0.5, self.dim)
        candidate = np.clip(self.population[self.global_best] + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def optimize(self, func):
        self.evaluate_population(func)
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if np.random.rand() < 0.15:
                    trial = self.adaptive_random_search()
                else:
                    target = self.population[i]
                    a, b, c = self.select_parents()
                    mutant = self.mutate(a, b, c, self.scaling_factor[i])
                    trial = self.crossover(target, mutant)

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.scaling_factor[i] = min(1.0, self.scaling_factor[i] * 1.1)  # Adaptive F factor increase
                    if trial_fitness < self.fitness[self.global_best]:
                        self.global_best = i
                        self.local_best[i] = trial

                if self.eval_count >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def __call__(self, func):
        best_solution, best_fitness = self.optimize(func)
        return best_solution, best_fitness