import numpy as np

class SwarmSynergisticQuantumAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, dim)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.annealing_rate = 0.99
        self.min_temp = 1e-3

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb)
        personal_best_positions = population.copy()
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_score = personal_best_scores[global_best_index]
        evaluations = self.population_size
        temperature = 1.0

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - population[i]) +
                                 self.social_coeff * r2 * (global_best_position - population[i]))
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

                # Quantum annealing inspired update
                if np.random.rand() < np.exp(-1.0 / max(temperature, self.min_temp)):
                    quantum_flip = np.random.normal(scale=temperature, size=self.dim) * (ub - lb)
                    population[i] = np.clip(population[i] + quantum_flip, lb, ub)

                score = func(population[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = population[i].copy()
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best_position = population[i].copy()
                        global_best_score = score

            temperature *= self.annealing_rate

        return global_best_position, global_best_score