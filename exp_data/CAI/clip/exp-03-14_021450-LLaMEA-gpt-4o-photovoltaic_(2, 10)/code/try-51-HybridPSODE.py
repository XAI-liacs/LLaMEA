import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w_initial = 0.9  # initial inertia weight
        self.w_final = 0.4    # final inertia weight
        self.f_initial = 0.5  # initial differential weight, changed from constant 0.8
        self.f_final = 0.9    # final differential weight
        self.cr = 0.9  # crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0
        
        while evaluations < self.budget:
            self.w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            self.f = self.f_initial + (self.f_final - self.f_initial) * (evaluations / self.budget)  # Adaptive differential weight

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i] if global_best_position is not None else 0))
                
                if np.random.rand() < 0.2:
                    velocities[i] = np.random.uniform(-1, 1, self.dim)
                
                candidate_position = population[i] + velocities[i]
                chaotic_factor = 4 * r1 * (1 - r1)  # Logistic map for perturbation
                candidate_position += chaotic_factor * (ub - lb) * 0.05
                candidate_position = np.clip(candidate_position, lb, ub)

                candidate_score = func(candidate_position)
                evaluations += 1

                if candidate_score < personal_best_scores[i]:
                    personal_best_scores[i] = candidate_score
                    personal_best_positions[i] = candidate_position

                if candidate_score < global_best_score:
                    global_best_score = candidate_score
                    global_best_position = candidate_position

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant_vector = np.clip(personal_best_positions[a] + self.f * (personal_best_positions[b] - personal_best_positions[c]), lb, ub)
                trial_vector = np.copy(personal_best_positions[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == jrand:
                        trial_vector[j] = mutant_vector[j]
                
                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector

                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector

        return global_best_position, global_best_score