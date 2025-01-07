import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive parameter
        self.c2 = 1.5  # social parameter
        self.F = 0.8  # DE scaling factor
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evaluations = self.population_size

        while evaluations < self.budget:
            # Update velocities and positions - PSO part
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.w = 0.3 + 0.7 * (1 - evaluations / self.budget)  # Adaptive inertia weight, tweaked range
            velocities = (self.w * velocities 
                          + self.c1 * r1 * (personal_best_positions - population) 
                          + self.c2 * r2 * (global_best_position - population))
            max_vel = (ub - lb) * 0.2  # Adaptive velocity clamping
            velocities = np.clip(velocities, -max_vel, max_vel)  # Change made here
            population = np.clip(population + velocities, lb, ub)

            # Evaluate new solutions
            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = population[i]
            new_global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[new_global_best_index] < global_best_score:
                global_best_score = personal_best_scores[new_global_best_index]
                global_best_position = personal_best_positions[new_global_best_index]

            # Differential Evolution step
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                self.F = 0.6 + 0.4 * np.random.rand()  # Adaptive scaling factor
                mutant_vector = np.clip(a + self.F * (b - c), lb, ub)
                population_diversity = np.std(population) / (ub - lb)
                crossover = np.random.rand(self.dim) < (self.CR * population_diversity)  # Adaptive crossover probability based on diversity
                trial_vector = np.where(crossover, mutant_vector, population[i])
                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < scores[i]:
                    population[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector

        return global_best_position, global_best_score