import numpy as np

class AdaptiveHarmonyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.harmony_memory_size = 20
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.f_min, self.f_max = self.bounds
        self.de_cr = 0.9  # Crossover rate for Differential Evolution
        self.de_f = 0.8   # Differential mutation factor

    def _initialize_harmony_memory(self):
        return np.random.uniform(self.f_min, self.f_max, (self.harmony_memory_size, self.dim))

    def __call__(self, func):
        eval_count = 0
        harmony_memory = self._initialize_harmony_memory()
        harmony_fitness = np.array([func(hm) for hm in harmony_memory])
        eval_count += self.harmony_memory_size

        while eval_count < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected_harmony = harmony_memory[np.random.randint(self.harmony_memory_size)]
                    new_harmony[i] = selected_harmony[i]
                    if np.random.rand() < self.par:
                        new_harmony[i] += np.random.uniform(-1, 1)
                        new_harmony[i] = np.clip(new_harmony[i], self.f_min, self.f_max)
                else:
                    new_harmony[i] = np.random.uniform(self.f_min, self.f_max)

            new_fitness = func(new_harmony)
            eval_count += 1

            if new_fitness < np.max(harmony_fitness):
                worst_index = np.argmax(harmony_fitness)
                harmony_memory[worst_index] = new_harmony
                harmony_fitness[worst_index] = new_fitness

            # Differential Evolution Refinement
            for i in range(self.harmony_memory_size):
                if eval_count >= self.budget:
                    break
                indices = list(range(self.harmony_memory_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = harmony_memory[a] + self.de_f * (harmony_memory[b] - harmony_memory[c])
                mutant = np.clip(mutant, self.f_min, self.f_max)
                trial = np.where(np.random.rand(self.dim) < self.de_cr, mutant, harmony_memory[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < harmony_fitness[i]:
                    harmony_memory[i] = trial
                    harmony_fitness[i] = trial_fitness

        best_index = np.argmin(harmony_fitness)
        return harmony_memory[best_index]