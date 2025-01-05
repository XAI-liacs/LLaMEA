import numpy as np

class AdaptiveQuantumMemetic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, dim)
        self.alpha = 0.9  # Initial global search rate
        self.beta = 0.1   # Initial local search rate
        self.local_search_intensity = 5
        self.global_search_intensity = 10
        self.mutation_rate = 0.05  # Mutation rate for memetic search
        self.quantum_influence = 0.1  # Quantum influence factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(ind) for ind in population])
        best_index = np.argmin(scores)
        best_solution = population[best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptively adjust global and local search rates
            self.alpha = 0.8 + 0.2 * (1 - evaluations / self.budget)
            self.beta = 0.2 + 0.8 * (evaluations / self.budget)

            # Quantum-inspired global search
            for _ in range(self.global_search_intensity):
                if evaluations >= self.budget:
                    break
                new_solution = best_solution + self.quantum_influence * np.random.normal(size=self.dim)
                new_solution = np.clip(new_solution, lb, ub)
                new_score = func(new_solution)
                evaluations += 1
                if new_score < scores[best_index]:
                    scores[best_index] = new_score
                    best_solution = new_solution

            # Memetic local search
            for _ in range(self.local_search_intensity):
                if evaluations >= self.budget:
                    break
                local_candidate = best_solution + self.mutation_rate * np.random.uniform(-1, 1, self.dim) * (ub - lb)
                local_candidate = np.clip(local_candidate, lb, ub)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[best_index]:
                    scores[best_index] = local_score
                    best_solution = local_candidate

        # Return the best solution found
        return best_solution, scores[best_index]