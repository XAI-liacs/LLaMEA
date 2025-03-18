import numpy as np
from scipy.stats import norm

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.f = 0.8
        self.cr = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = lb + (ub - lb) * norm.cdf(self.w * np.random.uniform(-1, 1, (self.population_size, self.dim)))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        personal_best_positions = population.copy()
        personal_best_scores = fitness.copy()
        global_best_position = population[np.argmin(fitness)]
        global_best_score = np.min(fitness)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            population_entropy = -np.sum(np.log(np.var(population, axis=0, ddof=1) + 1e-10))
            dynamic_w = 0.1 + (0.4 * (1 - evaluations / self.budget))  # Changed line
            dynamic_f = 0.5 + (0.3 * (evaluations / self.budget))  # Changed line

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                self.w = dynamic_w  # Changed line
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i]))
                candidate_position = population[i] + velocities[i]
                candidate_position = np.clip(candidate_position, lb, ub)
                candidate_fitness = func(candidate_position)
                evaluations += 1

                if evaluations < self.budget:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    self.f = dynamic_f  # Changed line
                    mutant_vector = a + self.f * (b - c)
                    mutant_vector = np.clip(mutant_vector, lb, ub)
                    
                    trial_vector = np.where(np.random.rand(self.dim) < self.cr, mutant_vector, candidate_position)
                    trial_fitness = func(trial_vector)
                    evaluations += 1

                    if trial_fitness < candidate_fitness:
                        candidate_position = trial_vector
                        candidate_fitness = trial_fitness

                if candidate_fitness < personal_best_scores[i]:
                    personal_best_positions[i] = candidate_position
                    personal_best_scores[i] = candidate_fitness

                if candidate_fitness < global_best_score:
                    global_best_position = candidate_position
                    global_best_score = candidate_fitness

                population[i] = candidate_position

        return global_best_position