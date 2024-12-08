import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        # Initialize particles and population
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        population = np.copy(positions)
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        scores = np.full(self.population_size, np.inf)

        # Evaluate initial particles and population
        for i in range(self.population_size):
            score = func(positions[i])
            personal_best_scores[i] = score
            scores[i] = score

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            # PSO Update
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            np.clip(positions, self.lower_bound, self.upper_bound, out=positions)

            # Evaluate new positions
            for i in range(self.population_size):
                new_score = func(positions[i])
                evaluations += 1
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = positions[i]

            # DE Mutation and Crossover
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = indices
                mutant = np.clip(population[a] + self.F * (population[b] - population[c]), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask | (np.random.randint(self.dim) == np.arange(self.dim)), mutant, population[i])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    scores[i] = trial_score
                    population[i] = trial

            # Update Global Best
            current_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_best_idx]
                global_best_position = personal_best_positions[current_best_idx]

            if evaluations >= self.budget:
                break

        return global_best_position, global_best_score