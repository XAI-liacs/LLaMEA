import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.inertia_weight = 0.7
        self.mutation_rate = 0.1
        self.de_mutation_factor = 0.8  # New parameter for DE
        self.de_crossover_rate = 0.7  # New parameter for DE

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(individual) for individual in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            diversity = np.std(population, axis=0).mean()
            self.inertia_weight *= 0.99 ** (1 + diversity) * ((self.budget - evaluations) / self.budget)  # Adjust inertia
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coefficient * r1 * (personal_best_positions - population) +
                          self.social_coefficient * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, lb, ub)
            
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    mutant_vector = np.clip(a + self.de_mutation_factor * (b - c), lb, ub)
                    crossover = np.random.rand(self.dim) < self.de_crossover_rate
                    trial_vector = np.where(crossover, mutant_vector, population[i])
                    population[i] = trial_vector

            scores = np.array([func(individual) for individual in population])
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = population[i]

            if np.min(scores) < global_best_score:
                global_best_idx = np.argmin(scores)
                global_best_position = population[global_best_idx]
                global_best_score = scores[global_best_idx]

        return global_best_position, global_best_score