import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.best_global_position = None
        self.best_global_score = np.inf
        self.F = 0.8  # Scaling factor for DE
        self.CR = 0.9  # Crossover probability for DE

    def evaluate_population(self, func):
        scores = np.array([func(p) for p in self.particles])
        for i, score in enumerate(scores):
            if score < self.best_personal_scores[i]:
                self.best_personal_scores[i] = score
                self.best_personal_positions[i] = self.particles[i]
            if score < self.best_global_score:
                self.best_global_score = score
                self.best_global_position = self.particles[i]
        return scores

    def update_velocities_and_positions(self):
        w = 0.5  # inertia weight
        c1 = 1.5  # cognitive parameter
        c2 = 1.5  # social parameter
        r1, r2 = np.random.rand(2)

        for i in range(self.population_size):
            cognitive_component = c1 * r1 * (self.best_personal_positions[i] - self.particles[i])
            social_component = c2 * r2 * (self.best_global_position - self.particles[i])
            self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)

    def differential_evolution_step(self):
        new_population = np.copy(self.particles)
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = np.clip(self.particles[a] + self.F * (self.particles[b] - self.particles[c]), -5.0, 5.0)
            crossover = np.random.rand(self.dim) < self.CR
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_vector = np.clip(trial_vector, -5.0, 5.0)
            if func(trial_vector) < func(self.particles[i]):
                new_population[i] = trial_vector
        self.particles = new_population

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate_population(func)
            self.update_velocities_and_positions()
            self.differential_evolution_step()
            evaluations += self.population_size
        return self.best_global_position, self.best_global_score