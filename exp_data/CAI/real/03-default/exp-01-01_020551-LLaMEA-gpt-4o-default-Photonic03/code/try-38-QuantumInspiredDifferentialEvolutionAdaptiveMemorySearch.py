import numpy as np

class QuantumInspiredDifferentialEvolutionAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 3 * dim)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.1  # Quantum superposition factor
        self.adaptive_rate = 0.01

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        global_best_index = np.argmin(scores)
        global_best_position = population[global_best_index].copy()

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                # Quantum-inspired perturbation
                if np.random.rand() < self.alpha:
                    q = np.random.normal(0, 1, self.dim)
                    trial += q * (global_best_position - trial)

                new_population[i] = np.clip(trial, lb, ub)
                new_score = func(new_population[i])
                evaluations += 1

                if new_score < scores[i]:
                    scores[i] = new_score
                    population[i] = new_population[i]
                    if new_score < scores[global_best_index]:
                        global_best_index = i
                        global_best_position = new_population[i].copy()

            # Adaptive parameter tuning
            self.F = max(0.1, self.F - self.adaptive_rate * (evaluations / self.budget))
            self.CR = min(1.0, self.CR + self.adaptive_rate * (evaluations / self.budget))

        return global_best_position, scores[global_best_index]