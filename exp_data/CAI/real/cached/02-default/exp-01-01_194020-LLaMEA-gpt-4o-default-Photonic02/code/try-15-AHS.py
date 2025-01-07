import numpy as np

class AHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.harmony_memory_size = 20
        self.harmonies = []
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjusting Rate
        self.bw = 0.01   # Bandwidth for pitch adjustment

    def initialize_harmonies(self, lb, ub):
        harmonies = []
        for _ in range(self.harmony_memory_size):
            harmony = lb + (ub - lb) * np.random.rand(self.dim)
            harmonies.append(harmony)
        return harmonies

    def pitch_adjustment(self, harmony, lb, ub):
        if np.random.rand() < self.par:
            dim_to_adjust = np.random.randint(self.dim)
            harmony[dim_to_adjust] += self.bw * (2 * np.random.rand() - 1)
            harmony[dim_to_adjust] = np.clip(harmony[dim_to_adjust], lb[dim_to_adjust], ub[dim_to_adjust])

    def generate_new_harmony(self, lb, ub):
        new_harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                harmony_index = np.random.randint(self.harmony_memory_size)
                new_harmony[i] = self.harmonies[harmony_index][i]
            else:
                new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        
        self.pitch_adjustment(new_harmony, lb, ub)
        return new_harmony

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.harmonies = self.initialize_harmonies(lb, ub)

        while evaluations < self.budget:
            new_harmony = self.generate_new_harmony(lb, ub)
            value = func(new_harmony)
            evaluations += 1

            if value < self.best_value:
                self.best_value = value
                self.best_solution = new_harmony.copy()

            # Replace worst harmony if the new harmony is better
            worst_idx = np.argmax([func(harmony) for harmony in self.harmonies])
            if value < func(self.harmonies[worst_idx]):
                self.harmonies[worst_idx] = new_harmony

            if evaluations >= self.budget:
                break

        return self.best_solution, self.best_value