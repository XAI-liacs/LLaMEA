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
        self.adaptive_factor = 0.9  # New adaptive factor
        self.elite_preservation_rate = 0.1  # New elite preservation rate

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

        while func_evals < self.budget:
            for i in range(self.population_size):
                if i < int(self.population_size * self.elite_preservation_rate):
                    continue  # Preserve elite solutions
                
                chaotic_sequence[i] = self.chaos_map(chaotic_sequence[i])
                adaptive_inertia = self.inertia_weight * (1 - func_evals / self.budget) * self.adaptive_factor
                velocity[i] = (
                    adaptive_inertia * velocity[i]
                    + self.cognitive_coeff * chaotic_sequence[i] * (personal_best[i] - pop[i])
                    + self.social_coeff * chaotic_sequence[i] * (global_best - pop[i])
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