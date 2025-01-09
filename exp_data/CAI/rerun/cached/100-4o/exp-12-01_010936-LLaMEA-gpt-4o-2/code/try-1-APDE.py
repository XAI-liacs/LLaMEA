import numpy as np

class APDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.pop = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.personal_best_positions = self.pop.copy()
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.9   # inertia weight adjusted for improved exploration
        self.F = 0.8   # DE scaling factor
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Particle Swarm Update
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.pop[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.pop[i]))
                self.pop[i] += self.velocities[i]
                self.pop[i] = np.clip(self.pop[i], -5, 5)
                
                # Evaluate new solution
                score = func(self.pop[i])
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.pop[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.pop[i].copy()

                if evaluations >= self.budget:
                    break

            # Differential Evolution Crossover and Mutation
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                idxs = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.pop[idxs]
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, -5, 5)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.pop[i])
                trial_score = func(trial)
                evaluations += 1

                if trial_score < self.personal_best_scores[i]:
                    self.pop[i] = trial
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial.copy()
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial.copy()

        return self.global_best_position, self.global_best_score