import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.inertia_weight = 0.5
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best[i] - population[i]) +
                                 self.social_const * r2 * (global_best - population[i]))
                particle = population[i] + velocities[i]
                particle = np.clip(particle, bounds[:, 0], bounds[:, 1])

                # Evaluate
                particle_score = func(particle)
                evaluations += 1
                if particle_score < personal_best_scores[i]:
                    personal_best[i] = particle
                    personal_best_scores[i] = particle_score
                    if particle_score < func(global_best):
                        global_best = particle

                # DE update
                if evaluations < self.budget:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                    trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])

                    # Evaluate trial
                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best[i] = trial
                        personal_best_scores[i] = trial_score
                        if trial_score < func(global_best):
                            global_best = trial

            # Update global best
            current_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_best_idx] < func(global_best):
                global_best = personal_best[current_best_idx]

        return global_best