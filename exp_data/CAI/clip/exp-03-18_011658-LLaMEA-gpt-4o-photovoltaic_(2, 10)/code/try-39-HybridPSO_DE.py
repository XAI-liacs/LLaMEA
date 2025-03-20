import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(budget / (dim * 10)))
        self.c1_initial = 2.0  # initial cognitive component
        self.c2_initial = 2.0  # initial social component
        self.c1_final = 0.5  # final cognitive component
        self.c2_final = 0.5  # final social component
        self.w = 0.5   # inertia weight
        self.diff_weight = 0.8  # differential weight for DE
        self.cr = 0.9   # crossover probability for DE

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
            w_dynamic = 0.4 + 0.1 * (1 - eval_count / self.budget)  # Dynamic inertia weight
            c1_dynamic = self.c1_final + (self.c1_initial - self.c1_final) * (1 - eval_count / self.budget)
            c2_dynamic = self.c2_final + (self.c2_initial - self.c2_final) * (1 - eval_count / self.budget)
            velocities = w_dynamic * velocities + c1_dynamic * r1 * (personal_best - pop) + c2_dynamic * r2 * (global_best - pop)
            velocities = np.clip(velocities, -abs(ub-lb), abs(ub-lb))  # Velocity clamping
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

            adaptive_diff_weight = self.diff_weight * (1 - eval_count / self.budget)
            adaptive_cr = self.cr * (eval_count / self.budget)
            switch_strategy = eval_count < self.budget * 0.5
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = x1 + adaptive_diff_weight * (x2 - x3)
                trial = np.where(np.random.rand(self.dim) < adaptive_cr, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                trial_score = func(trial)
                eval_count += 1
                if trial_score < scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score

        return global_best