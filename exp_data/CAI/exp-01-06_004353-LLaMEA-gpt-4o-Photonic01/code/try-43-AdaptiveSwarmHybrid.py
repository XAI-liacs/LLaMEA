import numpy as np

class AdaptiveSwarmHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros_like(pop)
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        eval_count = self.population_size
        
        while eval_count < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * np.cos(np.pi * eval_count / (2 * self.budget))**1.2  # Adaptive weight decay
            velocity_scale = 1 - eval_count / self.budget  # Dynamic scaling
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))
                amplification_factor = 1 / (1 + np.exp(-10 * (eval_count / self.budget - 0.5)))
                pop[i] += velocities[i] * amplification_factor * velocity_scale  # Apply scaling
                pop[i] = np.clip(pop[i], lb, ub)

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                dynamic_F = self.F - (self.F * eval_count / self.budget)
                mutant = np.clip(a + dynamic_F * (b - c), lb, ub)
                trial = np.copy(pop[i])
                dynamic_CR = self.CR * (1 - eval_count / self.budget) * (0.5 + 0.5 * np.cos(np.pi * eval_count / self.budget))  # Time-varying component
                for j in range(self.dim):
                    if np.random.rand() < dynamic_CR:
                        trial[j] = mutant[j]
                trial_score = func(trial)
                eval_count += 1
                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best = trial
                        global_best_score = trial_score

        return global_best, global_best_score