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
            if iteration % 10 == 0:  # Reduce population size iteratively
                self.pop_size = max(5, self.pop_size - 1)
                population = population[:self.pop_size]
                velocities = velocities[:self.pop_size]
                personal_best = personal_best[:self.pop_size]
                personal_best_scores = personal_best_scores[:self.pop_size]
                
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), bounds[:, 0], bounds[:, 1])
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])

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
                                 self.cognitive_const * r1 * (personal_best[i] - population[i]) +
                                 self.social_const * r2 * (global_best - population[i]))
                velocities[i] *= 0.9  # Damping factor for velocity
                population[i] += velocities[i]
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

                particle_score = func(population[i])
                evaluations += 1
                if particle_score < personal_best_scores[i]:
                    personal_best[i] = population[i]
                    personal_best_scores[i] = particle_score
                    if particle_score < func(global_best):
                        global_best = population[i]
            
            iteration += 1

            self.f = 0.5 + 0.4 * (evaluations / self.budget)
            self.cr = 0.6 + 0.4 * np.random.rand()

        return global_best