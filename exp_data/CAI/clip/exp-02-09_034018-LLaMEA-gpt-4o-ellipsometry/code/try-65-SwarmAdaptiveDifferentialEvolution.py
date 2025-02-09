import numpy as np

class SwarmAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover rate
        self.inertia_weight = 0.7
        self.cognitive_const = 1.4
        self.social_const = 1.4

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        
        evaluations = self.pop_size
        iteration = 0
        max_iterations = self.budget // self.pop_size
        
        while evaluations < self.budget:
            elite_idx = np.argmin(personal_best_scores)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), bounds[:, 0], bounds[:, 1])
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])

                if np.random.rand() < 0.1:
                    trial = (trial + personal_best[elite_idx]) / 2

                # Lévy flight-based exploration
                if np.random.rand() < 0.05:
                    levy = np.random.standard_cauchy(size=self.dim)
                    trial = np.clip(trial + levy, bounds[:, 0], bounds[:, 1])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < func(global_best):
                        global_best = trial

            inertia_weight_dynamic = 0.9 - (iteration / max_iterations) * 0.5
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight_dynamic * velocities[i] +
                                 (self.cognitive_const + 0.5 * np.cos(iteration / max_iterations * np.pi)) * r1 * (personal_best[i] - population[i]) +
                                 (self.social_const + 0.5 * np.cos(iteration / max_iterations * np.pi)) * r2 * (global_best - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])
                
                exploration_factor = np.sin(0.5 * iteration / max_iterations * np.pi)
                velocities[i] *= exploration_factor

                particle_score = func(population[i])
                evaluations += 1
                if particle_score < personal_best_scores[i]:
                    personal_best[i] = population[i]
                    personal_best_scores[i] = particle_score
                    if particle_score < func(global_best):
                        global_best = population[i]

            iteration += 1
            
            if iteration % 10 == 0:  # Hybridization step every 10 iterations
                self.f = 0.5 + 0.1 * np.sin(iteration / max_iterations * np.pi)

            self.f = 0.4 + 0.2 * np.random.rand()  # Adjusted stochastic mutation factor
            self.cr = 0.5 + 0.5 * np.random.rand()  # Dynamic crossover rate

        return global_best