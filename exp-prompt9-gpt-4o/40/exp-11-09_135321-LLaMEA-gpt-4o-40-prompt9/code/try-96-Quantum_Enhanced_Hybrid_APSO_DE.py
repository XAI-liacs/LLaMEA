import numpy as np

class Quantum_Enhanced_Hybrid_APSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Quantum-inspired parameters
        self.num_particles = 40
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.8
        self.social_coeff = 1.5
        self.quantum_coeff = 0.1  # New quantum potential factor

        # DE parameters
        self.F_base = 0.8  # Slightly increased scaling factor for exploration
        self.CR_base = 0.85

        # Initialization
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_inertia(self, iter_count):
        return 0.9 - iter_count * (0.6 / self.budget)

    def adaptive_mutation_strategy(self, current_iter, total_iters):
        return self.F_base + 0.2 * np.sin(np.pi * current_iter / total_iters)

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, L)
        v = np.random.normal(0, 1, L)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def quantum_operator(self, position):
        return self.quantum_coeff * np.random.randn(self.dim) * (self.global_best_position - position)

    def __call__(self, func):
        evals = 0
        iter_count = 0

        while evals < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_particles

            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            quantum_component = np.array([self.quantum_operator(pos) for pos in self.positions])
            self.velocities = (self.chaotic_inertia(iter_count) * self.velocities +
                               cognitive_component + social_component + quantum_component)
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            F = self.adaptive_mutation_strategy(iter_count, self.budget)
            for i in range(self.num_particles):
                idx1, idx2, idx3 = [np.random.randint(0, self.num_particles) for _ in range(3)]
                x1, x2, x3 = self.positions[idx1], self.positions[idx2], self.positions[idx3]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR_base, mutant_vector, self.positions[i])
                
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.01 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            evals += self.num_particles
            iter_count += 1

        return self.global_best_position, self.global_best_score