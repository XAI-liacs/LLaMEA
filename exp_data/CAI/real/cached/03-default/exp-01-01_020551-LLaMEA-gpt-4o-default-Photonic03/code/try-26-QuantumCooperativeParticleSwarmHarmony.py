import numpy as np

class QuantumCooperativeParticleSwarmHarmony:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, dim)
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.inertia = 0.7  # Inertia weight for PSO
        self.cognitive_coef = 1.5  # Cognitive coefficient
        self.social_coef = 1.5  # Social coefficient
        self.beta = 0.05  # Quantum-inspired learning rate
        self.cooperation_factor = 2.0  # Cooperation factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        personal_best_positions = harmony_memory.copy()
        personal_best_scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = harmony_memory[global_best_index].copy()
        velocities = np.random.uniform(-1, 1, (self.harmony_memory_size, self.dim)) * (ub - lb)
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            # Dynamic adjustments based on progress
            self.hmcr = 0.7 + 0.3 * (1 - evaluations / self.budget)
            self.par = 0.2 + 0.5 * (evaluations / self.budget)

            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[selected, i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-0.1, 0.1) * (ub[i] - lb[i])
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])

                # Quantum-inspired update
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=1)
                    new_harmony[i] = global_best_position[i] + q * (ub[i] - lb[i])

            new_score = func(new_harmony)
            evaluations += 1

            # Update personal and global bests if new harmony is better
            if new_score < personal_best_scores.max():
                worst_index = np.argmax(personal_best_scores)
                harmony_memory[worst_index] = new_harmony
                personal_best_scores[worst_index] = new_score
                personal_best_positions[worst_index] = new_harmony

            if new_score < personal_best_scores[global_best_index]:
                global_best_index = np.argmin(personal_best_scores)
                global_best_position = harmony_memory[global_best_index].copy()

            # Cooperative swarm dynamic update
            if evaluations < self.budget:
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities = (self.inertia * velocities +
                              self.cognitive_coef * r1 * (personal_best_positions - harmony_memory) +
                              self.social_coef * r2 * (global_best_position - harmony_memory) +
                              self.cooperation_factor * np.random.rand(self.harmony_memory_size, self.dim) * 
                              (global_best_position - harmony_memory.mean(axis=0)))

                harmony_memory = np.clip(harmony_memory + velocities, lb, ub)

        return global_best_position, personal_best_scores[global_best_index]