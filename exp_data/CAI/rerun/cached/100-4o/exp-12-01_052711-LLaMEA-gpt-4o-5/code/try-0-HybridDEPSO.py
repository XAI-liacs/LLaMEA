import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.initialize_population()
        self.velocities = np.zeros((self.population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.current_evaluations = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def __call__(self, func):
        while self.current_evaluations < self.budget:
            # Evaluate population
            for i in range(self.population_size):
                score = func(self.population[i])
                self.current_evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]

            # Adaptive strategy between DE and PSO
            if self.current_evaluations < self.budget / 2:
                self.differential_evolution_step(func)
            else:
                self.particle_swarm_optimization_step(func)

        return self.global_best_position

    def differential_evolution_step(self, func):
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

            mutant_vector = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
            trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, self.population[i])

            # Evaluate trial vector
            trial_score = func(trial_vector)
            self.current_evaluations += 1

            if trial_score < self.personal_best_scores[i]:
                self.population[i] = trial_vector
                self.personal_best_scores[i] = trial_score
                self.personal_best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

    def particle_swarm_optimization_step(self, func):
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive (personal) coefficient
        c2 = 1.5  # Social (global) coefficient

        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)

            self.velocities[i] = (
                w * self.velocities[i]
                + c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                + c2 * r2 * (self.global_best_position - self.population[i])
            )

            self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            # Evaluate updated particle
            score = func(self.population[i])
            self.current_evaluations += 1

            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]