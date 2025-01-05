import numpy as np

class QuantumHarmonyGPSO:
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
        self.memeplex_segments = 3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.harmony_memory_size, self.dim)) * (ub - lb)
        personal_best_positions = harmony_memory.copy()
        personal_best_scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = self.harmony_memory_size

        while evaluations < self.budget:
            self.hmcr = 0.7 + 0.3 * (1 - evaluations / self.budget)
            self.par = 0.2 + 0.5 * (evaluations / self.budget)
            
            # Dynamic memeplex segmentation
            indices = np.argsort(personal_best_scores)
            memeplexes = np.array_split(indices, self.memeplex_segments)

            # Generate new harmony vector with quantum-inspired exploration
            for memeplex in memeplexes:
                new_harmony = np.empty(self.dim)
                for i in range(self.dim):
                    if np.random.rand() < self.hmcr:
                        new_harmony[i] = harmony_memory[np.random.choice(memeplex), i]
                        if np.random.rand() < self.par:
                            new_harmony[i] += np.random.uniform(-0.1, 0.1) * (ub[i] - lb[i])
                    else:
                        new_harmony[i] = np.random.uniform(lb[i], ub[i])
                    
                    if np.random.rand() < self.beta:
                        q = np.random.normal(loc=0, scale=1)
                        new_harmony[i] = global_best_position[i] + q * (ub[i] - lb[i])

                # Evaluate new harmony
                new_score = func(new_harmony)
                evaluations += 1

                # Update personal and global best
                if new_score < personal_best_scores[memeplex[-1]]:
                    worst_index = memeplex[-1]
                    harmony_memory[worst_index] = new_harmony
                    personal_best_scores[worst_index] = new_score
                    personal_best_positions[worst_index] = new_harmony

                if new_score < personal_best_scores[global_best_index]:
                    global_best_index = worst_index
                    global_best_position = new_harmony

                if evaluations >= self.budget:
                    break

            # PSO update
            if evaluations < self.budget:
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities = (self.inertia * velocities +
                              self.c1 * r1 * (personal_best_positions - harmony_memory) +
                              self.c2 * r2 * (global_best_position - harmony_memory))
                harmony_memory = np.clip(harmony_memory + velocities, lb, ub)

        return global_best_position, personal_best_scores[global_best_index]