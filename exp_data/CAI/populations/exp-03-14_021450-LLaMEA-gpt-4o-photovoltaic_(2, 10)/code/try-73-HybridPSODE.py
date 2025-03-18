import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.final_population_size = 30  # New: adaptive population size
        self.c1_initial = 1.5
        self.c2_initial = 1.5
        self.w_initial = 0.9
        self.w_final = 0.4
        self.f_initial = 0.8
        self.cr_initial = 0.9
        self.velocity_bounds = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-self.velocity_bounds, self.velocity_bounds, (population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0
        
        while evaluations < self.budget:
            population_size = int(self.final_population_size + (self.initial_population_size - self.final_population_size) * (1 - evaluations / self.budget))  # New: adaptive population size
            self.w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
            self.f = self.f_initial * (1 - evaluations / self.budget)
            self.cr = self.cr_initial * np.random.rand()
            
            self.c1 = self.c1_initial + 0.5 * np.cos(2 * np.pi * evaluations / self.budget)
            self.c2 = self.c2_initial + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)

            for i in range(population_size):  # Changed: use adaptive population size
                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i] if global_best_position is not None else 0))

                if np.random.rand() < 0.15:
                    velocities[i] = np.random.uniform(-self.velocity_bounds, self.velocity_bounds, self.dim)

                candidate_position = population[i] + velocities[i]
                candidate_position = np.clip(candidate_position, lb, ub)
                candidate_score = func(candidate_position)
                evaluations += 1

                if candidate_score < personal_best_scores[i]:
                    personal_best_scores[i] = candidate_score
                    personal_best_positions[i] = candidate_position

                if candidate_score < global_best_score:
                    global_best_score = candidate_score
                    global_best_position = candidate_position

            for i in range(population_size):  # Changed: use adaptive population size
                if evaluations >= self.budget:
                    break
                
                indices = list(range(population_size))
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