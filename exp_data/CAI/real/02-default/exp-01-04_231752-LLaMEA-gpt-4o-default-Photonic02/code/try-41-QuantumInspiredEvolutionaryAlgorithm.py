import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.chromosomes = None
        self.best_chromosome = None
        self.best_score = np.inf
        self.alpha = np.pi / 4  # Rotation angle for quantum gates

    def _initialize_chromosomes(self, lb, ub):
        self.chromosomes = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
    
    def _quantum_rotation(self, chromosome, best_chromosome):
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                theta = self.alpha if chromosome[i] < best_chromosome[i] else -self.alpha
                chromosome[i] = chromosome[i] + theta * (best_chromosome[i] - chromosome[i])
        return chromosome

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_chromosomes(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.chromosomes[i])
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_chromosome = self.chromosomes[i].copy()

            # Perform quantum rotation for exploration and exploitation
            for i in range(self.population_size):
                self.chromosomes[i] = self._quantum_rotation(self.chromosomes[i], self.best_chromosome)
                self.chromosomes[i] = np.clip(self.chromosomes[i], self.lb, self.ub)

        return self.best_chromosome, self.best_score