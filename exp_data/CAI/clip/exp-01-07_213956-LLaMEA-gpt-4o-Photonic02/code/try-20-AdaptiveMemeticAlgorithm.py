import numpy as np

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Initial population size
        self.alpha = 0.5    # Local search probability factor
        self.beta = 0.1     # Adaptation rate for parameters
        self.mutation_strength = 0.1  # Initial mutation strength

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.pop_size
        convergence_rate = []

        while evaluations < self.budget:
            for i in range(self.pop_size):
                candidate = self.local_search(population[i], lb, ub, func)
                candidate_value = func(candidate)
                evaluations += 1

                if candidate_value < personal_best_values[i]:
                    personal_best[i] = candidate
                    personal_best_values[i] = candidate_value

                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value

                # Adaptive parameter update based on convergence
                if len(convergence_rate) > 5:  # Using a window of 5 evaluations
                    recent_improvements = np.diff(convergence_rate[-5:])
                    if np.all(recent_improvements < 0):
                        self.mutation_strength = max(self.mutation_strength * (1 + self.beta), 0.001)
                        self.alpha = min(self.alpha * (1 + self.beta), 1.0)
                    else:
                        self.mutation_strength = max(self.mutation_strength * (1 - self.beta), 0.001)
                        self.alpha = max(self.alpha * (1 - self.beta), 0.1)

                convergence_rate.append(global_best_value)
                if evaluations >= self.budget:
                    break

        return global_best

    def local_search(self, individual, lb, ub, func):
        if np.random.rand() < self.alpha:
            perturbation = np.random.normal(0, self.mutation_strength, self.dim)
            local_candidate = individual + perturbation
            local_candidate = np.clip(local_candidate, lb, ub)
            if func(local_candidate) < func(individual):
                return local_candidate
        return individual