import numpy as np

class QuantumEnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)  # Larger initial population based on dimension
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.beta = 0.3  # Quantum influence factor
        self.q_learning_rate = 0.05  # Quantum learning rate for adaptability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_index = np.argmin(scores)
        best_position = population[best_index].copy()

        while evaluations < self.budget:
            for j in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
  
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[j])

                # Quantum-inspired perturbation
                if np.random.rand() < self.beta:
                    q_step = np.random.normal(0, self.q_learning_rate, self.dim) * (ub - lb)
                    trial += q_step

                trial = np.clip(trial, lb, ub)
                trial_score = func(trial)
                evaluations += 1

                if trial_score < scores[j]:
                    population[j] = trial
                    scores[j] = trial_score

                if trial_score < scores[best_index]:
                    best_position = trial
                    best_index = j

            # Adapt mutation factor and crossover probability
            self.mutation_factor = 0.5 + 0.3 * np.random.rand()
            self.crossover_probability = 0.5 + 0.2 * np.random.rand()

        return best_position, scores[best_index]