import numpy as np

class EnhancedADHSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hms = 30
        self.hmcr = 0.9 + 0.1 * np.random.rand()
        self.par = 0.5
        self.bw = 0.01 * (self.upper_bound - self.lower_bound)
        self.evaluations = 0
        self.f = 0.8
        self.cr = 0.9
        self.learning_rate = 0.5  # New adaptive learning rate

    def random_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def pitch_adjustment(self, solution, best_solution):
        diversity_factor = np.std(solution) / self.dim
        adaptive_bw = self.bw * (0.5 + 0.5 * diversity_factor)
        for i in range(self.dim):
            if np.random.rand() < self.par:
                adjustment = adaptive_bw * (2 * np.random.rand() - 1)
                solution[i] = np.clip(solution[i] + adjustment, self.lower_bound, self.upper_bound)
        return solution

    def differential_mutation(self, target_idx, harmony_memory):
        idxs = list(range(self.hms))
        idxs.remove(target_idx)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        scaler = 0.5 + np.random.rand() * 0.5
        mutant = harmony_memory[a] + scaler * (harmony_memory[b] - harmony_memory[c])
        mutant = np.where(np.random.rand(self.dim) < self.cr, mutant, harmony_memory[target_idx])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def update_learning_rate(self, best_fitness, harmony_memory_fitness):
        # Adaptive learning rate based on fitness improvement
        self.learning_rate = 0.5 * (1 - best_fitness / max(harmony_memory_fitness))

    def __call__(self, func):
        harmony_memory = [self.random_solution() for _ in range(self.hms)]
        harmony_memory_fitness = [func(sol) for sol in harmony_memory]
        self.evaluations += self.hms

        best_solution = harmony_memory[np.argmin(harmony_memory_fitness)]
        best_fitness = min(harmony_memory_fitness)

        while self.evaluations < self.budget:
            self.update_learning_rate(best_fitness, harmony_memory_fitness)
            new_solution = np.zeros(self.dim)

            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected_harmony = harmony_memory[np.random.randint(self.hms)]
                    new_solution[i] = selected_harmony[i]
                else:
                    new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound)

            new_solution = self.pitch_adjustment(new_solution, best_solution)
            new_solution = self.differential_mutation(np.argmin(harmony_memory_fitness), harmony_memory)

            new_solution_fitness = func(new_solution)
            self.evaluations += 1

            if new_solution_fitness < max(harmony_memory_fitness):
                worst_index = np.argmax(harmony_memory_fitness)
                harmony_memory[worst_index] = new_solution
                harmony_memory_fitness[worst_index] = new_solution_fitness

                if new_solution_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_solution_fitness

        return best_solution