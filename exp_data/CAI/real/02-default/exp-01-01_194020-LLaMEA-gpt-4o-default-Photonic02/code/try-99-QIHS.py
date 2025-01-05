import numpy as np

class QIHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.harmony_size = 20
        self.harmonies = []

    def initialize_harmony_memory(self, lb, ub):
        harmony_memory = []
        for _ in range(self.harmony_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            harmony_memory.append({'position': position, 'value': float('inf')})
        return harmony_memory

    def update_harmony(self, harmony, global_best, lb, ub, harmony_consideration_rate, pitch_adjustment_rate):
        new_position = harmony['position'].copy()
        for i in range(self.dim):
            if np.random.rand() < harmony_consideration_rate:
                new_position[i] = np.random.choice([h['position'][i] for h in self.harmonies])
                if np.random.rand() < pitch_adjustment_rate:
                    phi = np.arccos(1 - 2 * np.random.rand())
                    direction = np.sign(np.random.rand() - 0.5)
                    new_position[i] += direction * np.tan(phi) * (global_best[i] - new_position[i])
            else:
                new_position[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

        new_position = np.clip(new_position, lb, ub)
        harmony['position'] = new_position

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.harmonies = self.initialize_harmony_memory(lb, ub)
        global_best = None
        global_best_value = float('inf')
        harmony_consideration_rate = 0.9
        pitch_adjustment_rate = 0.3

        while evaluations < self.budget:
            for harmony in self.harmonies:
                value = func(harmony['position'])
                evaluations += 1

                if value < harmony['value']:
                    harmony['value'] = value

                if value < global_best_value:
                    global_best_value = value
                    global_best = harmony['position'].copy()

                if evaluations >= self.budget:
                    break

            for harmony in self.harmonies:
                self.update_harmony(harmony, global_best, lb, ub, harmony_consideration_rate, pitch_adjustment_rate)

        return global_best, global_best_value