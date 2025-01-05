import numpy as np

class HarmonyAdaptiveQuantumSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.harmony_memory = np.random.uniform(size=(self.population_size, dim))
        self.fitness_scores = np.full(self.population_size, np.inf)
        self.best_harmony = None
        self.best_fitness = np.inf
        self.fitness_evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        harmony_memory_consideration_rate = 0.95
        pitch_adjustment_rate = 0.3
        bandwidth = 0.1

        while self.fitness_evaluations < self.budget:
            new_harmony = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < harmony_memory_consideration_rate:
                    idx = np.random.choice(self.population_size)
                    new_harmony[i] = self.harmony_memory[idx, i]
                    if np.random.rand() < pitch_adjustment_rate:
                        new_harmony[i] += bandwidth * np.random.uniform(-1.0, 1.0)
                else:
                    new_harmony[i] = lower_bound[i] + np.random.rand() * (upper_bound[i] - lower_bound[i])
            
            new_harmony = np.clip(new_harmony, lower_bound, upper_bound)
            new_fitness = func(new_harmony)
            self.fitness_evaluations += 1

            max_fitness_idx = np.argmax(self.fitness_scores)
            if new_fitness < self.fitness_scores[max_fitness_idx]:
                self.harmony_memory[max_fitness_idx] = new_harmony
                self.fitness_scores[max_fitness_idx] = new_fitness

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_harmony = new_harmony

        return self.best_harmony