import numpy as np

class ProbabilisticSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.harmony_memory = np.random.rand(self.harmony_memory_size, self.dim)
        self.velocities = np.zeros((self.harmony_memory_size, self.dim))
        self.harmony_memory_improv_rate = 0.85
        self.inertia_weight = 0.7
        self.personal_best_positions = self.harmony_memory.copy()
        self.global_best_position = None
        self.personal_best_scores = np.full(self.harmony_memory_size, float('inf'))
        self.global_best_score = float('inf')
        self.bandwidth = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.harmony_memory = lb + (ub - lb) * self.harmony_memory

        for _ in range(self.budget):
            for i in range(self.harmony_memory_size):
                # Generate new harmony using probabilistic approach
                new_harmony = np.copy(self.harmony_memory[i])
                if np.random.rand() < self.harmony_memory_improv_rate:
                    idx = np.random.choice(self.harmony_memory_size)
                    new_harmony += self.inertia_weight * self.velocities[i] + \
                                   self.bandwidth * (np.random.rand(self.dim) - 0.5)
                    new_harmony = np.clip(new_harmony, lb, ub)
                
                # Evaluate new harmony
                fitness = func(new_harmony)
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = new_harmony
                    if fitness < self.global_best_score:
                        self.global_best_score = fitness
                        self.global_best_position = new_harmony
                
                # Update velocities
                cognitive_component = np.random.rand(self.dim) * (self.personal_best_positions[i] - self.harmony_memory[i])
                social_component = np.random.rand(self.dim) * (self.global_best_position - self.harmony_memory[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component

            # Update harmony memory
            worst_idx = np.argmax([func(h) for h in self.harmony_memory])
            if func(new_harmony) < func(self.harmony_memory[worst_idx]):
                self.harmony_memory[worst_idx] = new_harmony

        return self.global_best_position