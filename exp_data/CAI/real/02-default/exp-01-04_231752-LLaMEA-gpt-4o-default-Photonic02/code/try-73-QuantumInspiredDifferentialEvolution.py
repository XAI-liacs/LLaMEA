import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.q_population = None  # Quantum population
        self.population = None  # Real population
        self.best_solution = None
        self.best_score = np.inf
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def _initialize_population(self, lb, ub):
        self.q_population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb

    def _quantum_rotation_gate(self, current, best, lb, ub):
        delta = (best - current) * np.random.rand(self.dim)
        theta = np.arctan2(delta, 1.0)
        qrotation = np.cos(theta) * current + np.sin(theta) * delta
        return np.clip(qrotation, lb, ub)

    def _mutate(self, idx, lb, ub):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, lb, ub)

    def _crossover(self, target, mutant):
        trial = np.where(np.random.rand(self.dim) < self.CR, mutant, target)
        return trial

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Quantum-inspired update
                self.q_population[i] = self._quantum_rotation_gate(self.q_population[i], self.best_solution or self.q_population[i], self.lb, self.ub)
                self.population[i] = self.q_population[i]

                # Apply differential evolution mutation and crossover
                mutant = self._mutate(i, self.lb, self.ub)
                trial = self._crossover(self.population[i], mutant)

                # Evaluate trial and update the best solution
                score = func(trial)
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = trial

                # Select between trial and current population
                if score < func(self.population[i]):
                    self.population[i] = trial

        return self.best_solution, self.best_score