import numpy as np

class QuantumGeneticHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3  # Pitch Adjustment Rate
        self.qbit_population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')
        self.evaluations = 0

    def quantum_to_real(self, qbit):
        return qbit * (self.ub - self.lb) + self.lb

    def _initialize_population(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        for i in range(self.population_size):
            real_position = self.quantum_to_real(self.qbit_population[i])
            score = func(real_position)
            self.fitness[i] = score
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest = real_position
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return

    def _crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) > 0.5
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def _harmony_search(self, func):
        for _ in range(self.population_size):
            if np.random.rand() < self.hmcr:
                idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
                new_qbit = self._crossover(self.qbit_population[idx1], self.qbit_population[idx2])
            else:
                new_qbit = np.random.rand(self.dim)
            
            if np.random.rand() < self.par:
                adjustment = (np.random.rand(self.dim) - 0.5) * 0.05
                new_qbit = (new_qbit + adjustment) % 1.0

            new_real = self.quantum_to_real(new_qbit)
            new_score = func(new_real)
            worst_idx = np.argmax(self.fitness)
            
            if new_score < self.fitness[worst_idx]:
                self.qbit_population[worst_idx] = new_qbit
                self.fitness[worst_idx] = new_score
                if new_score < self.gbest_score:
                    self.gbest = new_real
                    self.gbest_score = new_score
            
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return

    def __call__(self, func):
        self._initialize_population(func)
        while self.evaluations < self.budget:
            self._harmony_search(func)
        return self.gbest