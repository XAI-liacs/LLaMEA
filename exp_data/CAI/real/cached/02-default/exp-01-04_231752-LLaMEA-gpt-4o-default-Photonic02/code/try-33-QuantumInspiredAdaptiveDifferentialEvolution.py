import numpy as np

class QuantumInspiredAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.population = None
        self.best_individual = None
        self.best_score = np.inf
        self.memory = []
        self.memory_size = 5
        self.F = 0.5  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.alpha = 0.1  # Quantum influence factor
        
    def _initialize_population(self, lb, ub):
        # Quantum-inspired initialization
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        quantum_step = self.alpha * np.sin(np.pi * np.random.rand(self.population_size, self.dim))
        self.population += quantum_step * (ub - lb)
        self.population = np.clip(self.population, lb, ub)

    def _mutate(self, target_idx, lb, ub):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.F * (b - c)
        return np.clip(mutant, lb, ub)

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask): 
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                target = self.population[i]
                mutant = self._mutate(i, self.lb, self.ub)
                trial = self._crossover(target, mutant)

                score = func(trial)
                eval_count += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_individual = trial.copy()
                    # Store successful strategies
                    if len(self.memory) >= self.memory_size:
                        self.memory.pop(0)
                    self.memory.append({'F': self.F, 'CR': self.CR})

                if score < func(target):
                    self.population[i] = trial
                    if np.random.rand() < 0.3:  # Occasionally adapt F and CR
                        self.F = np.clip(self.F + np.random.uniform(-0.1, 0.1), 0.4, 0.9)
                        self.CR = np.clip(self.CR + np.random.uniform(-0.1, 0.1), 0.6, 1.0)

        return self.best_individual, self.best_score