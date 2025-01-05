import numpy as np

class AdaptiveSpiralDynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = max(10, dim)
        self.hmcr = 0.85
        self.par = 0.3
        self.spiral_factor = 0.5
        self.adaptive_memory_scale = 0.1
        self.mutation_strength = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.harmony_memory_size, self.dim))
        scores = np.array([func(harmony_memory[i]) for i in range(self.harmony_memory_size)])
        global_best_index = np.argmin(scores)
        global_best_position = harmony_memory[global_best_index].copy()
        evaluations = self.harmony_memory_size
        
        while evaluations < self.budget:
            new_harmony = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = harmony_memory[selected, i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-self.mutation_strength, self.mutation_strength) * (ub[i] - lb[i])
                        
                else:
                    new_harmony[i] = np.random.uniform(lb[i], ub[i])
             
                # Spiral Dynamic update
                if np.random.rand() < self.spiral_factor:
                    theta = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0, np.linalg.norm(ub - lb) / 2)
                    new_harmony[i] = global_best_position[i] + radius * np.cos(theta)
                    new_harmony[i] = np.clip(new_harmony[i], lb[i], ub[i])

            new_score = func(new_harmony)
            evaluations += 1
            
            # Adaptive Memory Replacement
            if new_score < scores.max():
                worst_index = np.argmax(scores)
                harmony_memory[worst_index] = new_harmony
                scores[worst_index] = new_score
                
            global_best_index = np.argmin(scores)
            global_best_position = harmony_memory[global_best_index].copy()
            
            # Adaptive memory refinement
            if evaluations < self.budget and np.random.rand() < self.adaptive_memory_scale:
                mutation_indices = np.random.choice(self.harmony_memory_size, size=int(self.harmony_memory_size * self.adaptive_memory_scale), replace=False)
                for index in mutation_indices:
                    mutation_vector = np.random.normal(0, self.mutation_strength, self.dim)
                    harmony_memory[index] = np.clip(harmony_memory[index] + mutation_vector * (ub - lb), lb, ub)
                    scores[index] = func(harmony_memory[index])
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

        return global_best_position, scores[global_best_index]