import numpy as np

class AQHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = min(50, budget // 5)  # Harmony memory size
        self.hmcr = 0.9  # Harmony memory consideration rate
        self.par = 0.3  # Pitch adjustment rate
        self.bandwidth = 0.1  # Bandwidth for pitch adjustment
        self.harmony_memory = None
        self.fitness_memory = None

    def initialize_harmony_memory(self, lb, ub):
        self.harmony_memory = np.random.uniform(lb, ub, (self.hms, self.dim))
        self.fitness_memory = np.full(self.hms, np.inf)

    def quantum_position_update(self, lb, ub):
        beta = np.random.normal(0, 1, self.dim)
        position = self.harmony_memory[np.argmin(self.fitness_memory)]
        new_position = position + beta * (np.random.uniform(lb, ub, self.dim) - position) * 0.1
        return np.clip(new_position, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_harmony_memory(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_harmony[j] = self.harmony_memory[np.random.randint(self.hms), j]
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.uniform(-self.bandwidth, self.bandwidth)
                else:
                    new_harmony[j] = np.random.uniform(lb[j], ub[j])
            
            new_harmony = np.clip(new_harmony, lb, ub)
            current_value = func(new_harmony)
            evaluations += 1
            
            if current_value < np.max(self.fitness_memory):
                worst_index = np.argmax(self.fitness_memory)
                self.harmony_memory[worst_index] = new_harmony
                self.fitness_memory[worst_index] = current_value

            if evaluations < self.budget:
                quantum_harmony = self.quantum_position_update(lb, ub)
                quantum_value = func(quantum_harmony)
                evaluations += 1

                if quantum_value < np.max(self.fitness_memory):
                    worst_index = np.argmax(self.fitness_memory)
                    self.harmony_memory[worst_index] = quantum_harmony
                    self.fitness_memory[worst_index] = quantum_value

        best_index = np.argmin(self.fitness_memory)
        return self.harmony_memory[best_index], self.fitness_memory[best_index]