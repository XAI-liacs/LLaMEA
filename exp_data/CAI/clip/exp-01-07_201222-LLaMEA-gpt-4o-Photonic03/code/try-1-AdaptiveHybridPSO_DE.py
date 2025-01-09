import numpy as np

class AdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 2.05
        self.c2 = 2.05
        self.w = 0.729
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.convergence_threshold = 1e-6
        self.adaptive_factor = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])

        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        evaluations = self.population_size
        last_global_best_score = global_best_score

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)

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

            # Adaptive adjustment based on convergence speed
            if abs(last_global_best_score - global_best_score) < self.convergence_threshold:
                self.w *= (1 + self.adaptive_factor)
                self.mutation_factor = min(self.mutation_factor * (1 + self.adaptive_factor), 1.0)
            else:
                self.w = max(self.w * (1 - self.adaptive_factor), 0.4)
                self.mutation_factor = max(self.mutation_factor * (1 - self.adaptive_factor), 0.5)
            
            last_global_best_score = global_best_score

            # Apply DE operations on a random subset of the population
            num_de_individuals = int(self.population_size * 0.3)
            de_indices = np.random.choice(self.population_size, num_de_individuals, replace=False)
            for idx in de_indices:
                idxs = [i for i in range(self.population_size) if i != idx]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, pop[idx])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[idx]:
                    scores[idx] = trial_score
                    pop[idx] = trial

        return global_best, global_best_score