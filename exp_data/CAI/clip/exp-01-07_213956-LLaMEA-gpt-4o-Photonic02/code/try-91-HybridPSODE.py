import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30  # Increased population size for diversity
        self.c1_start, self.c1_end = 2.5, 0.5  # Adaptive cognitive component
        self.c2_start, self.c2_end = 0.5, 2.5  # Adaptive social component
        self.w_start, self.w_end = 0.9, 0.4  # Adaptive inertia weight
        self.F = 0.4  # DE scaling factor
        self.CR = 0.8  # Reduced DE crossover probability

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
                curr_progress = evaluations / self.budget
                w = self.w_start - (self.w_start - self.w_end) * curr_progress
                c1 = self.c1_start - (self.c1_start - self.c1_end) * curr_progress
                c2 = self.c2_start + (self.c2_end - self.c2_start) * curr_progress

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (w * velocity[i] + 
                               c1 * r1 * (personal_best[i] - population[i]) +
                               c2 * r2 * (global_best - population[i]))
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