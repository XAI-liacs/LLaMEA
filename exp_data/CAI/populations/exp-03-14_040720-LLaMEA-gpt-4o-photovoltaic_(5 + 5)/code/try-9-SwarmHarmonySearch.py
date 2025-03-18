import numpy as np

class SwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.position_memory = []
        self.harmony_memory = []
        self.harmony_memory_size = 10
        self.hmcr = 0.9  # Harmony memory consideration rate
        self.par = 0.2   # Reduced Pitch adjustment rate
        self.alpha = 0.5 # Learning factor for swarm intelligence
        self.global_best = None
        self.global_best_value = float('inf')

    def initialize_memory(self, lb, ub):
        for _ in range(self.harmony_memory_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            self.harmony_memory.append(position)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        self.initialize_memory(lb, ub)

        func_evaluations = 0
        while func_evaluations < self.budget:
            new_position = self.generate_new_position(lb, ub)
            new_position_value = func(new_position)
            func_evaluations += 1

            if new_position_value < self.global_best_value:
                self.global_best = new_position
                self.global_best_value = new_position_value

            worst_index = np.argmax([func(pos) for pos in self.harmony_memory])
            if new_position_value < func(self.harmony_memory[worst_index]):
                self.harmony_memory[worst_index] = new_position

        return self.global_best

    def generate_new_position(self, lb, ub):
        new_position = np.zeros(self.dim)
        
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                selected = self.harmony_memory[np.random.randint(self.harmony_memory_size)]
                new_position[i] = selected[i]
                if np.random.rand() < self.par:
                    new_position[i] += self.alpha * (np.random.rand() - 0.5) * (ub[i] - lb[i])
            else:
                new_position[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

        return np.clip(new_position, lb, ub)