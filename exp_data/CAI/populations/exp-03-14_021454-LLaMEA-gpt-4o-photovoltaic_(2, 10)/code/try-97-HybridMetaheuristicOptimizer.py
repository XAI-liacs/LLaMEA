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

        elite_fraction = 0.1
        adaptive_lr = 0.1
        momentum = 0.9  

        while evaluations < self.budget:
            iteration_ratio = evaluations / self.budget
            self.cognitive_coefficient = 1.5 + 0.5 * iteration_ratio
            self.social_coefficient = 1.5 - 0.5 * iteration_ratio
            
            self.inertia_weight = 0.7 * (0.995 ** (1 + iteration_ratio))
            velocity_scaling = np.exp(-iteration_ratio)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coefficient * r1 * (personal_best_positions - population) +
                          self.social_coefficient * r2 * (global_best_position - population)) * velocity_scaling
            population += velocities
            population = np.clip(population, lb, ub)
            
            dynamic_mutation_rate = self.mutation_rate * (1 - iteration_ratio)
            mutation_scaling = 1 + 0.5 * iteration_ratio
            for i in range(self.population_size):
                if np.random.rand() < dynamic_mutation_rate:
                    historical_influence = global_best_position * 0.2
                    mutation_vector = np.random.uniform(lb, ub, self.dim) + historical_influence  # Updated line
                    population[i] = 0.5 * (population[i] + mutation_vector * mutation_scaling)

            top_elite_idx = np.argsort(personal_best_scores)[:int(elite_fraction * self.population_size)]
            for idx in top_elite_idx:
                population[idx] = personal_best_positions[idx] * momentum + population[idx] * (1 - momentum)

            scores = np.array([func(individual) for individual in population])
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = population[i]
                    adaptive_lr *= 1.02
                else:
                    adaptive_lr *= 0.98

            if np.min(scores) < global_best_score:
                global_best_idx = np.argmin(scores)
                global_best_position = population[global_best_idx]
                global_best_score = scores[global_best_idx]

        return global_best_position, global_best_score