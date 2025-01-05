import numpy as np

class QuantumCohortEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.cohort_size = max(10, dim)
        self.quantum_rate = 0.15  # Quantum adjustment probability
        self.mutation_rate = 0.1  # Mutation rate for exploration
        self.evolution_step_size = 0.05  # Step size for cohort evolution
        self.min_cohort_size = max(5, dim // 2)
        self.max_cohort_size = max(10, dim)
        self.cohort_adjustment_frequency = budget // 25

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        cohort = np.random.uniform(lb, ub, (self.cohort_size, self.dim))
        fitness = np.array([func(cohort[i]) for i in range(self.cohort_size)])
        best_index = np.argmin(fitness)
        best_position = cohort[best_index].copy()
        evaluations = self.cohort_size

        while evaluations < self.budget:
            candidate = np.empty(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.quantum_rate:
                    candidate[i] = best_position[i] + np.random.normal(0, 0.5) * (ub[i] - lb[i])
                else:
                    cohort_choice = np.random.randint(self.cohort_size)
                    candidate[i] = cohort[cohort_choice, i] + np.random.uniform(-self.evolution_step_size, self.evolution_step_size) * (ub[i] - lb[i])
                if np.random.rand() < self.mutation_rate:
                    candidate[i] += np.random.uniform(-self.mutation_rate, self.mutation_rate) * (ub[i] - lb[i])
            candidate = np.clip(candidate, lb, ub)

            candidate_score = func(candidate)
            evaluations += 1

            if candidate_score < fitness.max():
                worst_index = np.argmax(fitness)
                cohort[worst_index] = candidate
                fitness[worst_index] = candidate_score

            if candidate_score < fitness[best_index]:
                best_index = np.argmin(fitness)
                best_position = cohort[best_index].copy()

            if evaluations % self.cohort_adjustment_frequency == 0:
                self.cohort_size = self.min_cohort_size + (self.max_cohort_size - self.min_cohort_size) * (1 - evaluations / self.budget)
                self.cohort_size = int(np.clip(self.cohort_size, self.min_cohort_size, self.max_cohort_size))
                cohort = np.resize(cohort, (self.cohort_size, self.dim))
                fitness = np.resize(fitness, self.cohort_size)

        return best_position, fitness[best_index]