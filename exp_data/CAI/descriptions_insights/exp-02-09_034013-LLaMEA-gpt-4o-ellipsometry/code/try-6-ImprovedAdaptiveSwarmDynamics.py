import numpy as np

class ImprovedAdaptiveSwarmDynamics:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + int(2 * np.sqrt(dim))
        self.min_population_size = 5
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.inertia_initial = 0.9
        self.inertia_damp = 0.99
        self.max_velocity = 0.2
        self.dynamic_shrinkage = 0.95
        self.neighborhood_radius = max(1, int(0.1 * dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        positions = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        self.c1 = self.c1_initial
        self.c2 = self.c2_initial
        self.inertia = self.inertia_initial

        evaluations = 0

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

            # Determine neighborhood best position
            neighborhood_best_positions = []
            for i in range(population_size):
                neighborhood = np.argsort(np.linalg.norm(positions - positions[i], axis=1))[:self.neighborhood_radius]
                best_in_neighborhood = neighborhood[np.argmin(personal_best_scores[neighborhood])]
                neighborhood_best_positions.append(personal_best_positions[best_in_neighborhood])

            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                neighborhood_component = self.c2 * r2 * (neighborhood_best_positions[i] - positions[i])
                velocities[i] = (self.inertia * velocities[i] + cognitive_component + neighborhood_component)
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

            self.inertia *= self.inertia_damp
            self.c1 = max(0.5, self.c1_initial * (1 - evaluations / self.budget))
            self.c2 = max(0.5, self.c2_initial * (evaluations / self.budget))

            population_size = max(self.min_population_size, int(self.initial_population_size * (self.dynamic_shrinkage ** (evaluations / self.budget))))
            positions = positions[:population_size]
            velocities = velocities[:population_size]
            personal_best_positions = personal_best_positions[:population_size]
            personal_best_scores = personal_best_scores[:population_size]

        return global_best_position, global_best_score