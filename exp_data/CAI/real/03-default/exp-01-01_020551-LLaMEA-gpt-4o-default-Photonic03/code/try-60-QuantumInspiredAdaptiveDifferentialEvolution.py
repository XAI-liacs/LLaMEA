import numpy as np

class QuantumInspiredAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.f_cr = 0.9  # Crossover probability
        self.f_f = 0.8  # Differential weight
        self.beta = 0.3  # Quantum-inspired coefficient
        self.adaptive_rate = 0.01

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(individual) for individual in population])
        global_best_index = np.argmin(scores)
        global_best_position = population[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[idxs]
                mutant = a + self.f_f * (b - c)
                mutant = np.clip(mutant, lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.f_cr, mutant, population[i])
                
                # Quantum-inspired update
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=0.5)
                    trial += q * (global_best_position - trial)

                trial = np.clip(trial, lb, ub)
                trial_score = func(trial)
                evaluations += 1

                # Selection
                if trial_score < scores[i]:
                    new_population[i] = trial
                    scores[i] = trial_score
                else:
                    new_population[i] = population[i]

            population = new_population
            global_best_index = np.argmin(scores)
            global_best_position = population[global_best_index].copy()

            # Adaptive parameter tuning
            self.f_cr = max(0.5, self.f_cr - self.adaptive_rate * (evaluations / self.budget))
            self.f_f = min(1.0, self.f_f + self.adaptive_rate * (evaluations / self.budget))

        return global_best_position, scores[global_best_index]