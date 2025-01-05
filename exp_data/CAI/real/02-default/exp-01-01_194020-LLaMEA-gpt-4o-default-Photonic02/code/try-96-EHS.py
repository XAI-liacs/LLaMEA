import numpy as np

class EHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 25
        self.harmony_memory = []
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.freq_range = 0.1

    def initialize_harmony_memory(self, lb, ub):
        for _ in range(self.harmony_memory_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            value = float('inf')
            self.harmony_memory.append({'position': position, 'value': value})

    def update_harmony_memory(self, candidate):
        worst_harmony = max(self.harmony_memory, key=lambda x: x['value'])
        if candidate['value'] < worst_harmony['value']:
            self.harmony_memory.remove(worst_harmony)
            self.harmony_memory.append(candidate)

    def generate_new_harmony(self, lb, ub):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                selected_harmony = self.harmony_memory[np.random.randint(self.harmony_memory_size)]
                new_harmony[i] = selected_harmony['position'][i]
                if np.random.rand() < self.par:
                    new_harmony[i] += self.freq_range * (np.random.rand() - 0.5)
            else:
                new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        new_harmony = np.clip(new_harmony, lb, ub)
        return new_harmony

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.initialize_harmony_memory(lb, ub)

        while evaluations < self.budget:
            new_position = self.generate_new_harmony(lb, ub)
            new_value = func(new_position)
            evaluations += 1

            candidate = {'position': new_position, 'value': new_value}
            self.update_harmony_memory(candidate)

            if evaluations >= self.budget:
                break

        best_harmony = min(self.harmony_memory, key=lambda x: x['value'])
        return best_harmony['position'], best_harmony['value']