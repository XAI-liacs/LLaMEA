import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w = 0.7   # Inertia weight
        self.F = 0.5   # DE scaling factor
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                dynamic_w = self.w * (1 - evaluations / self.budget)
                dynamic_c1 = self.c1 * (1 - evaluations / self.budget)  # Modified line
                velocity[i] = (dynamic_w * velocity[i] + 
                               dynamic_c1 * r1 * (personal_best[i] - population[i]) +  # Modified line
                               self.c2 * r2 * (global_best - population[i]))
                candidate = population[i] + velocity[i]
                candidate = np.clip(candidate, lb, ub)

                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    dynamic_F = self.F * (1 + evaluations / self.budget)
                    mutation_vector = a + dynamic_F * (b - c)
                    mutation_vector = np.clip(mutation_vector, lb, ub)
                    crossover = np.random.rand(self.dim) < self.CR
                    candidate[crossover] = mutation_vector[crossover]

                candidate_value = func(candidate)
                evaluations += 1

                if candidate_value < personal_best_values[i]:
                    personal_best[i] = candidate
                    personal_best_values[i] = candidate_value

                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value

                if evaluations >= self.budget:
                    break

        return global_best