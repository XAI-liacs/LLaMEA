import numpy as np

class EnhancedDifferentialEvolutionDynamic:
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

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1
                if self.global_best is None or self.fitness[i] < self.fitness[self.global_best]:
                    self.global_best = i

    def select_parents(self):
        indices = np.random.choice(self.pop_size, 4, replace=False)  # Using 4 parents for enhanced mutation
        return self.population[indices]

    def mutate(self, a, b, c, d, F=0.5):  # Simplified constant F factor
        mutant = np.clip(a + F * (b - c + d - a), self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant, CR=0.9):  # Increased CR for enhanced exploration
        crossover_mask = np.random.rand(self.dim) < CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def adaptive_random_search(self):
        perturbation = np.random.uniform(-0.3, 0.3, self.dim)  # Slightly adjusted perturbation range
        candidate = np.clip(self.population[self.global_best] + perturbation, self.lower_bound, self.upper_bound)
        return candidate

    def competitive_selection(self, trial, trial_fitness, target_idx):
        if trial_fitness < self.fitness[target_idx]:
            return trial, trial_fitness
        else:
            return self.population[target_idx], self.fitness[target_idx]

    def optimize(self, func):
        self.evaluate_population(func)
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if np.random.rand() < 0.3:  # Increased chance to use adaptive random search
                    trial = self.adaptive_random_search()
                else:
                    target = self.population[i]
                    a, b, c, d = self.select_parents()
                    mutant = self.mutate(a, b, c, d)
                    trial = self.crossover(target, mutant)

                trial_fitness = func(trial)
                self.eval_count += 1

                self.population[i], self.fitness[i] = self.competitive_selection(trial, trial_fitness, i)

                if self.fitness[i] < self.fitness[self.global_best]:
                    self.global_best = i

                if self.eval_count >= self.budget:
                    break

            # Dynamically adjust population size based on eval_count
            if self.eval_count < self.budget * 0.5:
                self.pop_size = min(self.pop_size * 2, 20 * self.dim)  # Increase population size in early stages
                self.population = np.vstack((self.population, np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size - len(self.population), self.dim))))
                self.fitness = np.hstack((self.fitness, np.full(self.pop_size - len(self.fitness), np.inf)))

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def __call__(self, func):
        best_solution, best_fitness = self.optimize(func)
        return best_solution, best_fitness