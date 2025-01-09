import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20  # Initial population size
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.w = 0.7   # Inertia weight
        self.F = 0.5   # DE scaling factor
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.initial_pop_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.initial_pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.initial_pop_size
        pop_size = self.initial_pop_size

        while evaluations < self.budget:
            for i in range(pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                dynamic_w = self.w * (1 - evaluations / self.budget)  # Modified line
                velocity[i] = (dynamic_w * velocity[i] + 
                               self.c1 * r1 * (personal_best[i] - population[i]) + 
                               self.c2 * r2 * (global_best - population[i]))
                candidate = population[i] + velocity[i]
                candidate = np.clip(candidate, lb, ub)

                # Opposition-based learning
                opposition = lb + ub - candidate
                opposition_value = func(opposition)
                evaluations += 1

                if opposition_value < func(candidate):
                    candidate = opposition
                    candidate_value = opposition_value
                else:
                    candidate_value = func(candidate)

                # Differential Evolution mutation and crossover
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    dynamic_F = self.F * (1 + evaluations / self.budget)  # Modified line
                    mutation_vector = a + dynamic_F * (b - c)
                    mutation_vector = np.clip(mutation_vector, lb, ub)
                    crossover = np.random.rand(self.dim) < self.CR
                    candidate[crossover] = mutation_vector[crossover]

                if candidate_value < personal_best_values[i]:
                    personal_best[i] = candidate
                    personal_best_values[i] = candidate_value

                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value
                
                # Reduce population size adaptively
                if evaluations % (self.budget // 4) == 0 and pop_size > 5:
                    pop_size -= 1
                    population = population[:pop_size]
                    velocity = velocity[:pop_size]
                    personal_best = personal_best[:pop_size]
                    personal_best_values = personal_best_values[:pop_size]

                if evaluations >= self.budget:
                    break

        return global_best