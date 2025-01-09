import numpy as np

class EnhancedCoevolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.c1 = 2.0
        self.c2 = 1.5
        self.w = 0.4
        self.mutation_factor = 0.97  # Increased mutation factor slightly from 0.95 to 0.97
        self.crossover_rate = 0.85
        self.subpop_size_factor = 0.5
        self.adaptive_factor = 0.3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])

        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)

            velocities = self.w * velocities + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = pop + velocities
            pop = np.clip(pop, lb, ub)

            scores = np.array([func(ind) for ind in pop])
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best[i] = pop[i]

            if np.min(personal_best_scores) < global_best_score:
                global_best_score = np.min(personal_best_scores)
                global_best = personal_best[np.argmin(personal_best_scores)]

            self.subpop_size = int(self.population_size * self.subpop_size_factor)
            indices = np.arange(self.population_size)
            np.random.shuffle(indices)
            subpop1_indices = indices[:self.subpop_size]
            subpop2_indices = indices[self.subpop_size:]
            
            for idx in subpop1_indices:
                idxs = [i for i in subpop1_indices if i != idx]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation = self.mutation_factor * np.exp(-self.adaptive_factor * (evaluations / self.budget)**1.15)
                mutant = np.clip(a + adaptive_mutation * (b - c), lb, ub)
                dynamic_crossover_rate = self.crossover_rate * (1 - evaluations / self.budget)
                trial = np.where(np.random.rand(self.dim) < dynamic_crossover_rate, mutant, pop[idx])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[idx]:
                    scores[idx] = trial_score
                    pop[idx] = trial

            for idx in subpop2_indices:
                levy_step = np.random.normal(0, 0.1, self.dim) * (ub - lb) * (np.abs(np.random.standard_cauchy(self.dim)))
                candidate = pop[idx] + levy_step
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                evaluations += 1
                if candidate_score < scores[idx]:
                    scores[idx] = candidate_score
                    pop[idx] = candidate

        return global_best, global_best_score