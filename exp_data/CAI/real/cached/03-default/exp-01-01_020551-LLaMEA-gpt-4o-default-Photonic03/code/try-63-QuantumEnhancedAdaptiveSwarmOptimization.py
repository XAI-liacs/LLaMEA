import numpy as np

class QuantumEnhancedAdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(30, 10 * dim)
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.5  # inertia weight
        self.q_factor = 0.1  # quantum factor for enhanced exploration

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb)
        personal_best_positions = population.copy()
        personal_best_scores = np.array([func(indiv) for indiv in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i]))
                quantum_jump = np.random.normal(0, self.q_factor, self.dim)  # Quantum step
                velocities[i] += quantum_jump * (ub - lb)
                
                population[i] += velocities[i]
                population[i] = np.clip(population[i], lb, ub)
                
                current_score = func(population[i])
                evaluations += 1

                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = population[i].copy()
                    personal_best_scores[i] = current_score

                    if current_score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = personal_best_positions[i].copy()

        return global_best_position, personal_best_scores[global_best_index]