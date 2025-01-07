import numpy as np

class QuantumEnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 10 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.scale_factor = 0.5
        self.crossover_rate = 0.9
        self.evaluations = 0
        self.adaptive_threshold = 0.1

    def quantum_perturbation(self, position, factor=0.1):
        quantum_noise = np.random.normal(0, factor, self.dim)
        return np.clip(position + quantum_noise, 0, 1)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        
        for i in range(self.population_size):
            self.fitness[i] = func(self.positions[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.positions[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_solution

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.positions[indices]
                mutant = np.clip(a + self.scale_factor * (b - c), lb, ub)
                trial = np.copy(self.positions[i])
                
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial[cross_points] = mutant[cross_points]

                trial_fitness = func(trial)
                if trial_fitness < self.fitness[i]:
                    self.positions[i] = trial
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

                if np.random.rand() < self.adaptive_threshold:
                    self.positions[i] = self.quantum_perturbation(self.positions[i])

                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

            self._adapt_population()

        return self.best_solution

    def _adapt_population(self):
        if self.evaluations % (self.budget // 5) == 0:
            self.scale_factor = np.random.uniform(0.4, 0.9)
            self.crossover_rate = np.random.uniform(0.7, 0.95)