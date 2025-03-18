import numpy as np

class ChaoticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.chaos_map = lambda x: 4 * x * (1 - x)  # Logistic map for chaotic behavior
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_index = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_index]
        func_evals = self.population_size
        chaotic_sequence = np.random.rand(self.population_size)
        
        adaptive_inertia = self.inertia_weight  # Added line

        while func_evals < self.budget:
            for i in range(self.population_size):
                chaotic_sequence[i] = self.chaos_map(chaotic_sequence[i])
                adaptive_chaos = chaotic_sequence[i] * (func_evals / self.budget)  # Changed line

                velocity[i] = (
                    adaptive_inertia * velocity[i]  # Changed line
                    + self.cognitive_coeff * adaptive_chaos * (personal_best[i] - pop[i])  # Changed line
                    + self.social_coeff * adaptive_chaos * (global_best - pop[i])
                )
                pop[i] += velocity[i]
                pop[i] = np.clip(pop[i], lb, ub)

                score = func(pop[i])
                func_evals += 1
                if score < personal_best_scores[i]:
                    personal_best[i] = pop[i]
                    personal_best_scores[i] = score
                    if score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best = pop[i]

                if func_evals >= self.budget:
                    break

        return global_best