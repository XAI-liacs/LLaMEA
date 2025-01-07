import numpy as np

class EnhancedQuantumHarmonyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, dim)
        self.hmcr = 0.9  # Initial Harmony Memory Consideration Rate
        self.par = 0.3   # Initial Pitch Adjustment Rate
        self.f = 0.8     # Differential weight
        self.cr = 0.9    # Crossover probability
        self.inertia = 0.7  # Inertia weight for PSO
        self.c1 = 1.5       # Cognitive coefficient
        self.c2 = 1.5       # Social coefficient
        self.beta = 0.05    # Quantum-inspired learning rate
        self.swarms = 3     # Number of sub-swarms
        self.memory_split = np.array_split(range(self.harmony_memory_size), self.swarms)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.harmony_memory_size, self.dim)) * (ub - lb)
        personal_best_positions = harmony_memory.copy()
        personal_best_scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = harmony_memory[global_best_idx].copy()
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            self.hmcr = 0.7 + 0.3 * (1 - evaluations / self.budget)
            self.par = 0.2 + 0.5 * (evaluations / self.budget)
            self.beta = 0.05 + 0.45 * (evaluations / self.budget)

            for swarm_indices in self.memory_split:
                local_best_idx = swarm_indices[np.argmin(personal_best_scores[swarm_indices])]
                local_best_position = harmony_memory[local_best_idx].copy()

                for idx in swarm_indices:
                    new_harmony = np.empty(self.dim)
                    for i in range(self.dim):
                        if np.random.rand() < self.hmcr:
                            new_harmony[i] = harmony_memory[np.random.randint(self.harmony_memory_size), i]
                            if np.random.rand() < self.par:
                                new_harmony[i] += np.random.uniform(-0.1, 0.1) * (ub[i] - lb[i])
                        else:
                            new_harmony[i] = np.random.uniform(lb[i], ub[i])

                        if np.random.rand() < self.beta:
                            q = np.random.normal(loc=0, scale=1)
                            new_harmony[i] = global_best_position[i] + q * (ub[i] - lb[i])

                    if evaluations + 1 < self.budget:
                        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                        velocities[idx] = (self.inertia * velocities[idx] +
                                           self.c1 * r1 * (personal_best_positions[idx] - harmony_memory[idx]) +
                                           self.c2 * r2 * (local_best_position - harmony_memory[idx]))
                        harmony_memory[idx] = np.clip(harmony_memory[idx] + velocities[idx], lb, ub)

                    new_score = func(new_harmony)
                    evaluations += 1

                    if new_score < personal_best_scores[idx]:
                        personal_best_scores[idx] = new_score
                        personal_best_positions[idx] = new_harmony

                    if new_score < personal_best_scores[global_best_idx]:
                        global_best_idx = idx
                        global_best_position = new_harmony

        return global_best_position, personal_best_scores[global_best_idx]