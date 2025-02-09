import numpy as np

class QuantumInspiredAdaptiveParticleDynamics:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 15 + int(2.5 * np.sqrt(dim))
        self.min_population_size = 8
        self.quantum_superposition_factor = 0.5
        self.c1_initial = 1.5
        self.c2_initial = 1.5
        self.inertia_initial = 0.8
        self.inertia_damp = 0.98
        self.max_velocity = 0.3
        self.dynamic_shrinkage = 0.9

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

            # Quantum-inspired superposition and adaptive dynamics
            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.c2 * r2 * (global_best_position - positions[i])
                velocities[i] = (self.inertia * velocities[i] + cognitive_component + social_component)
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Quantum superposition factor to introduce stochastic perturbations
                if np.random.rand() < self.quantum_superposition_factor:
                    positions[i] = np.random.uniform(lb, ub, self.dim)

            # Dynamic adjustment of inertia, population size, and cognitive/social coefficients
            self.inertia *= self.inertia_damp
            self.c1 = max(0.4, self.c1_initial * (1 - evaluations / self.budget))
            self.c2 = max(0.4, self.c2_initial * (evaluations / self.budget))
            
            # Reduce population size over time to intensify search
            population_size = max(self.min_population_size, int(self.initial_population_size * (self.dynamic_shrinkage ** (evaluations / self.budget))))
            positions = positions[:population_size]
            velocities = velocities[:population_size]
            personal_best_positions = personal_best_positions[:population_size]
            personal_best_scores = personal_best_scores[:population_size]

        return global_best_position, global_best_score