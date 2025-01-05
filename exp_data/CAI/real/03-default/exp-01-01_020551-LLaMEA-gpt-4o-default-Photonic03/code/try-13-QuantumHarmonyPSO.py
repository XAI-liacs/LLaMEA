import numpy as np

class QuantumHarmonyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_harmony_memory_size = 10 + dim
        self.hmcr = 0.9  # Initial Harmony Memory Consideration Rate
        self.par = 0.3   # Initial Pitch Adjustment Rate
        self.f = 0.8     # Differential weight
        self.cr = 0.9    # Crossover probability
        self.inertia = 0.7  # Inertia weight for PSO
        self.c1 = 1.5       # Cognitive coefficient
        self.c2 = 1.5       # Social coefficient
        self.beta = 0.05    # Quantum-inspired learning rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = self.initial_harmony_memory_size
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        velocities = np.random.uniform(-1, 1, (harmony_memory_size, self.dim)) * (ub - lb)
        personal_best_positions = harmony_memory.copy()
        personal_best_scores = np.array([func(harmony_memory[i]) for i in range(harmony_memory_size)])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = harmony_memory_size

        while evaluations < self.budget:
            for i in range(harmony_memory_size):
                # Adaptive parameter adjustment
                self.hmcr = 0.7 + 0.3 * (1 - evaluations / self.budget)
                self.par = 0.2 + 0.5 * (evaluations / self.budget)

                # Generate new harmony vector with quantum-inspired exploration
                new_harmony = np.empty(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        new_harmony[j] = harmony_memory[np.random.randint(harmony_memory_size), j]
                        if np.random.rand() < self.par:
                            new_harmony[j] += np.random.uniform(-0.1, 0.1) * (ub[j] - lb[j])
                    else:
                        new_harmony[j] = np.random.uniform(lb[j], ub[j])

                    # Quantum-inspired update
                    if np.random.rand() < self.beta:
                        q = np.random.normal(loc=0, scale=1)
                        new_harmony[j] = global_best_position[j] + q * (ub[j] - lb[j])

                # Evaluate new harmony
                new_score = func(new_harmony)
                evaluations += 1

                # Update personal and global best asynchronously
                if new_score < personal_best_scores[i]:
                    personal_best_positions[i] = new_harmony
                    personal_best_scores[i] = new_score

                if new_score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = new_harmony

                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - harmony_memory[i]) +
                                 self.c2 * r2 * (global_best_position - harmony_memory[i]))
                harmony_memory[i] = np.clip(harmony_memory[i] + velocities[i], lb, ub)

            # Dynamically adjust memory size to balance exploration and exploitation
            if evaluations > self.budget / 2 and harmony_memory_size > self.dim:
                harmony_memory_size -= 1

        # Return the best solution found
        return global_best_position, personal_best_scores[global_best_index]