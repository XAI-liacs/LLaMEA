import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(budget / (dim * 10))) + 1
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.diff_weight = 0.8
        self.cr = 0.9

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
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            elite = np.argpartition(personal_best_scores, 2)[:2]  # Select top elites
            elite_pop = personal_best[elite]
            w_dynamic = 0.5 + 0.2 * np.sin(eval_count / self.budget * np.pi)  # Adaptive inertia weight
            self.c1, self.c2 = 1.5 + 0.5 * np.cos(eval_count / self.budget * np.pi), 1.5 - 0.5 * np.sin(eval_count / self.budget * np.pi)
            velocities = w_dynamic * velocities + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = np.clip(pop + velocities, lb, ub)

            scores = np.array([func(ind) for ind in pop])
            eval_count += self.population_size
            better_mask = scores < personal_best_scores
            personal_best[better_mask] = pop[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]

            min_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_idx] < global_best_score:
                global_best_score = personal_best_scores[min_idx]
                global_best = personal_best[min_idx]

            adaptive_diff_weight = self.diff_weight * (1 - eval_count / (1.5 * self.budget))
            adaptive_cr = self.cr * (1 - np.cos(eval_count / self.budget * np.pi))
            switch_strategy = eval_count < self.budget * 0.5
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = x1 + adaptive_diff_weight * (x2 - x3)
                trial = np.where(np.random.rand(self.dim) < adaptive_cr, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                trial_score = func(trial)
                eval_count += 1
                if trial_score < scores[i] or (i in elite and trial_score < personal_best_scores[i]):
                    pop[i] = trial
                    scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best = trial
                        global_best_score = trial_score

        return global_best