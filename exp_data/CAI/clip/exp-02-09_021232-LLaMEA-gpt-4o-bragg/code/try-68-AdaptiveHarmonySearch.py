import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9
        self.PAR = 0.3
        self.bandwidth = 0.05
        self.num_harmonies = 10
        self.harmony_memory = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.elite_rate = 0.2
        self.last_improvement = 0
        self.populations = 2  # New: Number of populations
        self.mutation_factor = 0.1  # New: Mutation factor

    def initialize_harmony_memory(self, bounds):
        self.harmony_memory = np.random.uniform(bounds.lb, bounds.ub, (self.num_harmonies, self.dim))
        self.harmony_fitness = np.array([self.evaluate(lambda x: np.sum(x), h) for h in self.harmony_memory])

    def evaluate(self, func, harmony):
        fitness = func(harmony)
        if fitness < self.best_fitness:
            self.last_improvement = 0
            self.best_fitness = fitness
            self.best_solution = harmony.copy()
        else:
            self.last_improvement += 1
        return fitness

    def update_harmony_memory(self, new_harmony, new_fitness):
        worst_idx = np.argmax(self.harmony_fitness)
        if new_fitness < self.harmony_fitness[worst_idx]:
            self.harmony_memory[worst_idx] = new_harmony
            self.harmony_fitness[worst_idx] = new_fitness

    def generate_new_harmony(self, bounds):
        new_harmony = np.zeros(self.dim)
        elite_harmonies = self.harmony_memory[np.argsort(self.harmony_fitness)][:int(self.elite_rate * self.num_harmonies)]
        for i in range(self.dim):
            if np.random.rand() < self.HMCR:
                if np.random.rand() < 0.5:
                    selected_harmony = elite_harmonies[np.random.randint(len(elite_harmonies))]
                    new_harmony[i] = selected_harmony[i]
                else:
                    new_harmony[i] = np.mean(self.harmony_memory[:, i])
                if np.random.rand() < self.PAR:
                    new_harmony[i] += np.random.normal(0, self.bandwidth)
            else:
                new_harmony[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])
        if np.random.rand() < self.mutation_factor:  # New: Mutation step
            new_harmony += np.random.normal(0, self.bandwidth, self.dim)
        return np.clip(new_harmony, bounds.lb, bounds.ub)

    def adjust_parameters(self):
        self.bandwidth *= np.exp(np.random.uniform(-0.02, 0.02))
        self.HMCR = min(1.0, max(0.7, self.HMCR + np.random.uniform(-0.01, 0.01)))
        self.PAR = min(1.0, max(0.1, self.PAR - 0.01 * self.last_improvement))
        self.elite_rate = max(0.1, min(0.5, self.elite_rate + np.random.uniform(-0.01, 0.01)))
        self.num_harmonies = max(5, int(self.num_harmonies * (1 + 0.01 * self.last_improvement)))  # Change

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_harmony_memory(bounds)

        evaluations = 0
        while evaluations < self.budget:
            for _ in range(self.populations):  # New: Multiple populations
                new_harmony = self.generate_new_harmony(bounds)
                new_fitness = self.evaluate(func, new_harmony)
                self.update_harmony_memory(new_harmony, new_fitness)
                self.adjust_parameters()
            evaluations += self.populations

        return self.best_solution