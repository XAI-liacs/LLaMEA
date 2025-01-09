import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.mutation_factor = 0.5  # Adaptively adjusted
        self.crossover_rate = 0.9   # Adaptively adjusted
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.pop_size, dim)
        )
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def chaos_local_search(self, solution):
        # Introduce chaotic sequence for local perturbation
        chaotic_seq = np.sin(np.linspace(0, np.pi, self.dim))
        perturbation = chaotic_seq * (self.upper_bound - self.lower_bound) * 0.01
        return np.clip(solution + perturbation, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        # Initial fitness evaluation
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # DE/rand/1/bin mutation strategy
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                
                # Binomial crossover
                trial = np.where(
                    np.random.rand(self.dim) < self.crossover_rate,
                    mutant,
                    self.population[i]
                )

                # Evaluate trial vector and apply chaos-based local search if beneficial
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
                else:
                    # Chaotic local search
                    local_trial = self.chaos_local_search(trial)
                    local_fitness = func(local_trial)
                    self.evaluations += 1
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_trial
                        self.fitness[i] = local_fitness
                        if local_fitness < self.best_fitness:
                            self.best_fitness = local_fitness
                            self.best_solution = local_trial

                # Adaptively adjust mutation factor and crossover rate
                self.mutation_factor = 0.5 + 0.3 * np.random.rand()
                self.crossover_rate = 0.9 * (1 - self.evaluations / self.budget)

        return self.best_solution