import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.mutation_factor = 0.5  # DE mutation factor
        self.crossover_prob = 0.9  # DE crossover probability
        self.alpha = 0.1  # Quantum-inspired parameter

    def quantum_inspired_mutation(self, target_idx):
        # A quantum-inspired mutation to increase diversity
        r1, r2, r3 = np.random.choice([i for i in range(self.population_size) if i != target_idx], 3, replace=False)
        mutant_vector = self.positions[r1] + self.mutation_factor * (self.positions[r2] - self.positions[r3])
        # Apply quantum-inspired influence
        mutant_vector += self.alpha * np.random.normal(size=self.dim)
        return mutant_vector

    def _update_individual(self, idx, func):
        mutant_vector = self.quantum_inspired_mutation(idx)
        trial_vector = np.copy(mutant_vector)
        
        for j in range(self.dim):
            if np.random.rand() > self.crossover_prob:
                trial_vector[j] = self.positions[idx, j]

        trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
        trial_score = func(trial_vector)

        if trial_score < self.pbest_scores[idx]:
            self.pbest_positions[idx] = trial_vector
            self.pbest_scores[idx] = trial_score

        if trial_score < self.gbest_score:
            self.gbest = trial_vector
            self.gbest_score = trial_score

        self.evaluations += 1

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.pbest_positions[i] = self.positions[i]
            self.pbest_scores[i] = score
            if score < self.gbest_score:
                self.gbest = self.positions[i]
                self.gbest_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._update_individual(i, func)
                if self.evaluations >= self.budget:
                    break

        return self.gbest