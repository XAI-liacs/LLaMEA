import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        personal_best_positions = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        eval_count = self.population_size
        stagnation_counter = 0
        stagnation_threshold = 10

        while eval_count < self.budget:
            self.w = 0.5 + 0.3 * np.cos(np.pi * eval_count / self.budget)  # Adjusted inertia weight adaptation
            self.c1 = 1.1 + (1.6 - 1.1) * (eval_count / self.budget)  # Adjusted cognitive coefficient
            self.c2 = 1.0 + (2.0 - 1.0) * (1 - eval_count / self.budget)  # Adjusted social coefficient
            
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - pop[i]) +
                                 self.c2 * r2 * (global_best_position - pop[i]))
                velocities[i] = np.clip(velocities[i], -0.2 * (ub - lb), 0.2 * (ub - lb))  # Adjusted velocity clipping
                if np.random.rand() < 0.4:  # Adaptive velocity scaling
                    velocities[i] *= (1 + 0.5 * np.random.rand())
                pop[i] = np.clip(pop[i] + velocities[i], lb, ub)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[indices]
                self.F = 0.7 + 0.3 * np.random.rand()  # Adjusted differential weight
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                neighborhood_factor = np.mean(pop, axis=0) + 0.3 * np.random.randn(self.dim)  # More diverse neighborhood-based mutation
                mutant = 0.6 * (mutant + neighborhood_factor)  # Adjusted combination factor
                cross_points = np.random.rand(self.dim) < (self.CR - 0.4 * eval_count / self.budget)
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_score = func(trial)
                eval_count += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best_position = trial
                        global_best_score = trial_score
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                    if stagnation_counter >= stagnation_threshold:
                        pop[i] = np.random.uniform(lb, ub, self.dim)
                        stagnation_counter = 0

        return global_best_position, global_best_score