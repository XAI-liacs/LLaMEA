import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.alpha = 0.5  # adaptive factor

    def quantum_position_update(self, particle, personal_best, global_best):
        beta = np.random.rand(self.dim)
        velocity_scale = self.alpha  # Changed line
        position = (1 - beta) * personal_best + beta * global_best
        return position + velocity_scale * np.random.uniform(-1, 1, size=self.dim)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = population.copy()
        scores = np.array([func(ind) for ind in population])
        personal_best_scores = scores.copy()

        evaluations = self.population_size
        best_idx = scores.argmin()
        global_best_position = population[best_idx].copy()
        global_best_score = scores[best_idx]

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - population[i])
                social_component = self.c2 * r2 * (global_best_position - population[i])

                inertia_weight = 0.9 - (0.8 * evaluations / self.budget) # Changing line
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                velocities[i] = np.clip(velocities[i], -1, 1)

                if np.random.rand() < 0.5:
                    particle_position = population[i] + velocities[i]
                else:
                    particle_position = self.quantum_position_update(population[i], personal_best_positions[i], global_best_position)

                particle_position = np.clip(particle_position, lb, ub)
                particle_score = func(particle_position)
                evaluations += 1

                self.alpha = max(0.1, 0.5 + 0.4 * np.cos(np.pi * evaluations / self.budget)) # Changed line

                if particle_score < personal_best_scores[i]:
                    personal_best_positions[i] = particle_position
                    personal_best_scores[i] = particle_score
                    if particle_score < global_best_score:
                        global_best_position = particle_position.copy()
                        global_best_score = particle_score

                if evaluations >= self.budget:
                    break

            self.c1 = 2.0 - 1.5 * (evaluations / self.budget) # Changed line
            self.c2 = 0.5 + 1.5 * (evaluations / self.budget) # Changed line
        
        return global_best_position, global_best_score