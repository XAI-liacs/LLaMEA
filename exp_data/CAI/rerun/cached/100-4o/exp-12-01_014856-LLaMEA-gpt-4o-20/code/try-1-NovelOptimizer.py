import numpy as np

class NovelOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(np.sqrt(self.budget))  # Adaptive population size
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Evaluate current candidate
                fitness = func(self.population[i])
                self.evaluations += 1

                # Update personal and global bests
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best[i] = self.population[i].copy()

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best = self.population[i].copy()

            # Adaptive randomness control (influence of differential evolution)
            F = np.random.uniform(0.5, 1.0)
            CR = np.random.uniform(0.1, 0.3)

            # Update particles using DE and PSO concepts
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Differential Evolution
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, self.population[i])

                # Particle Swarm Optimization velocity update
                inertia = 0.5
                cognitive = 2.0 * np.random.rand(self.dim) * (self.personal_best[i] - self.population[i])
                social = 2.0 * np.random.rand(self.dim) * (self.global_best - self.population[i])
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social

                # Position update
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Evaluate new candidate
                fitness = func(trial)
                self.evaluations += 1

                # Selection
                if fitness < self.personal_best_scores[i]:
                    self.population[i] = trial
                    self.personal_best_scores[i] = fitness
                    self.personal_best[i] = trial.copy()

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best = trial.copy()

        return self.global_best