import numpy as np

class QuantumDrivenCoevolutionaryHS_AF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 50
        self.harmony_memory = np.random.uniform(size=(self.harmony_memory_size, dim))
        self.fitness_evaluations = 0
        self.best_harmony = None
        self.best_fitness = np.inf
        self.archive = []

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        while self.fitness_evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < 0.7:  # Harmony memory consideration rate
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                else:
                    new_harmony[i] = lower_bound[i] + np.random.rand() * (upper_bound[i] - lower_bound[i])
                
                if np.random.rand() < 0.3:  # Pitch adjustment rate
                    new_harmony[i] += np.random.uniform(-0.05, 0.05) * (upper_bound[i] - lower_bound[i])
                    new_harmony[i] = np.clip(new_harmony[i], lower_bound[i], upper_bound[i])

            fitness = func(new_harmony)
            self.fitness_evaluations += 1

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_harmony = new_harmony.copy()

            if len(self.archive) < self.harmony_memory_size:
                self.archive.append(new_harmony.copy())
            else:
                worst_idx = np.argmax([func(h) for h in self.archive])
                if fitness < func(self.archive[worst_idx]):
                    self.archive[worst_idx] = new_harmony.copy()

            quantum_jump_prob = 0.2 - 0.1 * (self.fitness_evaluations / self.budget)
            if np.random.rand() < quantum_jump_prob:
                quantum_exploration = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                self.harmony_memory[np.random.randint(self.harmony_memory_size)] = quantum_exploration

            self.harmony_memory = np.array(sorted(self.harmony_memory, key=lambda x: func(x)))
            self.harmony_memory[-1] = new_harmony

        return self.best_harmony