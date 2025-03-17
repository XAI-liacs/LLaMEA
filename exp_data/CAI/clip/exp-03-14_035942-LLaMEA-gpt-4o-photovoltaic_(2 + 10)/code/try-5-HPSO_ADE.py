import numpy as np

class HPSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.9  # inertia weight
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        eval_count = self.population_size

        while eval_count < self.budget:
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            w_dynamic = 0.4 + (0.9 - 0.4) * (eval_count / self.budget)
            vel = w_dynamic * vel + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)

            pop = np.clip(pop + vel, lb, ub)

            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + self.mutation_factor * (pop[b] - pop[c]), lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(cross_points, mutant, pop[i])
                trial_score = func(trial)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial
                    personal_best_scores[i] = trial_score

                    if trial_score < global_best_score:
                        global_best = trial
                        global_best_score = trial_score

            elite_idx = np.argmin(personal_best_scores)
            pop = np.vstack((pop, personal_best[elite_idx]))
            pop = pop[np.random.choice(pop.shape[0], self.population_size, replace=False)]

        return global_best