import numpy as np

class EnhancedCoevolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_factor = 0.9
        self.crossover_rate = 0.8
        self.subpop_size = int(self.population_size / 2)
        self.adaptive_factor = 0.5
        
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

            # Adaptive inertia weight
            self.w = 0.9 - 0.5 * (evaluations / self.budget)

            # Update velocities and positions for PSO
            velocities = self.w * velocities + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = pop + velocities
            pop = np.clip(pop, lb, ub)

            # Evaluate new population
            scores = np.array([func(ind) for ind in pop])
            evaluations += self.population_size
            
            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best[i] = pop[i]
            
            if np.min(personal_best_scores) < global_best_score:
                global_best_score = np.min(personal_best_scores)
                global_best = personal_best[np.argmin(personal_best_scores)]

            # Divide population into two subpopulations
            indices = np.arange(self.population_size)
            np.random.shuffle(indices)
            subpop1_indices = indices[:self.subpop_size]
            subpop2_indices = indices[self.subpop_size:]
            
            # Apply enhanced DE on subpop1
            for idx in subpop1_indices:
                idxs = [i for i in subpop1_indices if i != idx]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation = self.mutation_factor * np.exp(-self.adaptive_factor * evaluations / self.budget)
                mutant = np.clip(a + adaptive_mutation * (b - c), lb, ub)
                # Adaptive crossover rate
                trial = np.where(np.random.rand(self.dim) < (self.crossover_rate * (1 - evaluations / self.budget)), mutant, pop[idx])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[idx]:
                    scores[idx] = trial_score
                    pop[idx] = trial

            # Apply adaptive local search on subpop2
            for idx in subpop2_indices:
                candidate = pop[idx] + np.random.normal(0, 0.1 * (1 - evaluations / self.budget), self.dim) * (ub - lb)
                candidate = np.clip(candidate, lb, ub)
                candidate_score = func(candidate)
                evaluations += 1
                if candidate_score < scores[idx]:
                    scores[idx] = candidate_score
                    pop[idx] = candidate

        return global_best, global_best_score