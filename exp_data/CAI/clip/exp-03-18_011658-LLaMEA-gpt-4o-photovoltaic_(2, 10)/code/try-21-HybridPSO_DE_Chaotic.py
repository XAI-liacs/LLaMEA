import numpy as np

class HybridPSO_DE_Chaotic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(budget / (dim * 10)))
        self.c1 = 2.0  # increased cognitive component
        self.c2 = 2.0  # increased social component
        self.w = 0.3   # reduced initial inertia weight
        self.diff_weight = 0.9  # adjusted differential weight for DE
        self.cr = 0.85  # adjusted crossover probability for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = personal_best_scores.min()
        eval_count = self.population_size

        while eval_count < self.budget:
            # PSO Update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            w_dynamic = 0.5 / (1 + eval_count / self.budget)  # More adaptive inertia weight
            velocities = w_dynamic * velocities + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = np.clip(pop + velocities, lb, ub)

            # Evaluate and update personal bests
            scores = np.array([func(ind) for ind in pop])
            eval_count += self.population_size
            better_mask = scores < personal_best_scores
            personal_best[better_mask] = pop[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]

            # Update global best
            min_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_idx] < global_best_score:
                global_best_score = personal_best_scores[min_idx]
                global_best = personal_best[min_idx]

            # DE Mutation and Crossover with chaotic local search
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = x1 + self.diff_weight * (x2 - x3)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                trial_score = func(trial)
                eval_count += 1
                if trial_score < scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score

        return global_best