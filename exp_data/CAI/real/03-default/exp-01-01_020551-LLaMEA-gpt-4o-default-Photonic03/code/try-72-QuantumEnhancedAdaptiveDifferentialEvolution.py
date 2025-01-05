import numpy as np

class QuantumEnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 5 * dim)  # A larger population for diversity
        self.scaling_factor = 0.8  # Differential weight
        self.crossover_rate = 0.9  # Crossover probability
        self.quantum_rate = 0.2  # Quantum perturbation frequency
        self.adaptive_rate = 0.1  # For adaptive mutation strategy

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(individual) for individual in population])
        global_best_index = np.argmin(scores)
        global_best_position = population[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scaling_factor * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Quantum perturbation
                if np.random.rand() < self.quantum_rate:
                    q = np.random.normal(loc=0, scale=0.5, size=self.dim)
                    trial += q * (ub - lb)
                    trial = np.clip(trial, lb, ub)

                trial_score = func(trial)
                evaluations += 1

                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

                    if trial_score < scores[global_best_index]:
                        global_best_index = i
                        global_best_position = trial

            # Adaptive mutation strategy
            if evaluations < self.budget:
                self.scaling_factor = 0.5 + 0.5 * (1 - evaluations / self.budget)

        return global_best_position, scores[global_best_index]