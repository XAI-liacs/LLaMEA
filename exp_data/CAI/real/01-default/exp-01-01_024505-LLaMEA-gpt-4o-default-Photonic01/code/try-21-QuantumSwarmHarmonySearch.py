import numpy as np

class QuantumSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = min(30, budget // 10)
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.mutation_rate = 0.05
        self.generational_progress = 0

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.harmony_memory_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.best_fitness:
            self.best_fitness = fitness[best_index]
            self.best_solution = self.population[best_index]
        return fitness

    def improve_harmony(self, lb, ub):
        new_harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                new_harmony[i] = self.population[np.random.randint(self.harmony_memory_size), i]
                if np.random.rand() < self.par:
                    pitch_adjustment = (ub[i] - lb[i]) * (np.random.rand() - 0.5)
                    new_harmony[i] += pitch_adjustment
            else:
                new_harmony[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
        return np.clip(new_harmony, lb, ub)

    def mutate_population(self, lb, ub):
        mutation_matrix = np.random.rand(self.harmony_memory_size, self.dim) < self.mutation_rate
        random_values = lb + (ub - lb) * np.random.rand(self.harmony_memory_size, self.dim)
        self.population = np.where(mutation_matrix, random_values, self.population)
        return np.clip(self.population, lb, ub)

    def quantum_tunneling(self, lb, ub):
        if np.random.rand() < 0.1:
            random_individuals = lb + (ub - lb) * np.random.rand(2, self.dim)
            indices = np.random.choice(self.harmony_memory_size, size=2, replace=False)
            self.population[indices] = random_individuals

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += len(fitness)

            if evaluations >= self.budget:
                break

            new_harmony = self.improve_harmony(lb, ub)
            new_fitness = func(new_harmony)
            evaluations += 1

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_harmony

            if new_fitness < max(fitness):
                worst_index = np.argmax(fitness)
                self.population[worst_index] = new_harmony

            self.mutate_population(lb, ub)
            self.quantum_tunneling(lb, ub)

            self.generational_progress = evaluations

        return self.best_solution, self.best_fitness