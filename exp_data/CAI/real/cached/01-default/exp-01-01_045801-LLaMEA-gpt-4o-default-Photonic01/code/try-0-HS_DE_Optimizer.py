import numpy as np

class HS_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.HMCR = 0.9  # Harmony Memory Consideration Rate
        self.PAR = 0.3   # Pitch Adjustment Rate
        self.memory_size = 10
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.memory = None
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_memory(self, lb, ub):
        self.memory = np.random.uniform(lb, ub, (self.memory_size, self.dim))
        for i in range(self.memory_size):
            fitness = self.evaluate(self.memory[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.memory[i].copy()

    def evaluate(self, solution):
        return self.func(solution)
    
    def harmony_search(self):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.HMCR:
                idx = np.random.randint(self.memory_size)
                new_harmony[i] = self.memory[idx, i]
                if np.random.rand() < self.PAR:
                    new_harmony[i] += np.random.uniform(-0.1, 0.1)  # Small adjustment
            else:
                new_harmony[i] = np.random.uniform(self.lb[i], self.ub[i])
        return new_harmony

    def differential_evolution(self):
        idxs = np.random.choice(self.memory_size, 3, replace=False)
        a, b, c = self.memory[idxs]
        mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
        trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, a)
        return trial_vector

    def __call__(self, func):
        self.func = func
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        evaluations = 0

        # Initialize memory
        self.initialize_memory(self.lb, self.ub)

        while evaluations < self.budget:
            # Hybrid step: Use Harmony Search and Differential Evolution
            if np.random.rand() < 0.5:
                candidate = self.harmony_search()
            else:
                candidate = self.differential_evolution()

            # Evaluate candidate solution
            candidate_fitness = self.evaluate(candidate)
            evaluations += 1
            
            # Update best solution and memory
            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_solution = candidate.copy()

            # Replace worst harmony in memory if better
            worst_idx = np.argmax([self.evaluate(h) for h in self.memory])
            if candidate_fitness < self.evaluate(self.memory[worst_idx]):
                self.memory[worst_idx] = candidate

        return {'solution': self.best_solution, 'fitness': self.best_fitness}