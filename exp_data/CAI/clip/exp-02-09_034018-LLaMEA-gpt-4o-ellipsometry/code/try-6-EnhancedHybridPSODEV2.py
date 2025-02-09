import numpy as np

class EnhancedHybridPSODEV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.inertia_weight = 0.9
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.elitism_rate = 0.1
        self.no_improvement_tolerance = 10  # Tolerance before reducing inertia

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]

        evaluations = self.pop_size
        no_improvement_counter = 0
        last_improvement = evaluations

        while evaluations < self.budget:
            self.cognitive_const = 1.5 + (0.5 * (evaluations / self.budget))
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best[i] - population[i]) +
                                 self.social_const * r2 * (global_best - population[i]))
                particle = population[i] + velocities[i]
                particle = np.clip(particle, bounds[:, 0], bounds[:, 1])

                particle_score = func(particle)
                evaluations += 1
                if particle_score < personal_best_scores[i]:
                    personal_best[i] = particle
                    personal_best_scores[i] = particle_score
                    if particle_score < func(global_best):
                        global_best = particle
                        no_improvement_counter = 0
                        last_improvement = evaluations
                else:
                    no_improvement_counter += 1

                if evaluations < self.budget:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
                    trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])

                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best[i] = trial
                        personal_best_scores[i] = trial_score
                        if trial_score < func(global_best):
                            global_best = trial
                            no_improvement_counter = 0
                            last_improvement = evaluations

            elite_count = int(self.elitism_rate * self.pop_size)
            elite_indices = np.argsort(personal_best_scores)[:elite_count]
            for idx in elite_indices:
                if personal_best_scores[idx] < func(global_best):
                    global_best = personal_best[idx]

            self.inertia_weight = max(0.4, 0.9 - (evaluations - last_improvement) / (self.budget / 10))

        return global_best