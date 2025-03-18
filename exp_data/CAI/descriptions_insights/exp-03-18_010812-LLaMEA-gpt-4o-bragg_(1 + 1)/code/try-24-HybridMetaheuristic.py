import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Adjusted population size with dynamic scaling
        self.population_size = max(5 * dim, 10 + int(dim * np.log(dim)))
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.f = 0.8
        self.cr = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        personal_best_positions = population.copy()
        personal_best_scores = fitness.copy()
        global_best_position = population[np.argmin(fitness)]
        global_best_score = np.min(fitness)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                self.w = 0.1 + 0.4 * (self.budget - evaluations) / self.budget
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i]))
                candidate_position = population[i] + velocities[i]
                # Introduced elite learning from global best
                elite_influence = 0.05 * (global_best_position - candidate_position)
                candidate_position += elite_influence
                candidate_position = np.clip(candidate_position, lb, ub)
                candidate_fitness = func(candidate_position)
                evaluations += 1

                if evaluations < self.budget:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    self.f = 0.5 + (0.3 * evaluations / self.budget)
                    mutant_vector = a + self.f * (b - c)
                    mutant_vector = np.clip(mutant_vector, lb, ub)
                    
                    self.cr = 0.8 * (self.budget - evaluations) / self.budget + 0.2 
                    trial_vector = np.where(np.random.rand(self.dim) < self.cr, mutant_vector, candidate_position)
                    trial_fitness = func(trial_vector)
                    evaluations += 1

                    if trial_fitness < candidate_fitness:
                        candidate_position = trial_vector
                        candidate_fitness = trial_fitness
                        self.c1 = 1.8 - (1.3 * evaluations / self.budget)

                if candidate_fitness < personal_best_scores[i]:
                    personal_best_positions[i] = candidate_position
                    personal_best_scores[i] = candidate_fitness

                if candidate_fitness < global_best_score:
                    global_best_position = candidate_position
                    global_best_score = candidate_fitness

                population[i] = candidate_position

        return global_best_position