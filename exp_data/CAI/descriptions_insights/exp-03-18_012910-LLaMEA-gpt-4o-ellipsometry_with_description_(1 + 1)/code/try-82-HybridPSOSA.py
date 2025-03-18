import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(30, int(budget / 50))
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.92

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        neighborhood_radius = 0.1  # Added neighborhood influence
        evaluations = self.population_size

        while evaluations < self.budget:
            for i, particle in enumerate(particles):
                inertia = self.inertia_weight * velocities[i]
                cognitive = self.cognitive_constant * np.random.rand(self.dim) * (personal_best_positions[i] - particle)
                social = self.social_constant * np.random.rand(self.dim) * (global_best_position - particle)
                velocities[i] = (inertia + cognitive + social + 0.05 * np.random.uniform(-2, 2, self.dim)) * self.temperature
                particles[i] = np.clip(particle + velocities[i], lb, ub)
                
                # Introduced neighborhood influence
                neighborhood_best_position = self.find_neighborhood_best(particles, personal_best_scores, i, neighborhood_radius)
                velocities[i] += 0.1 * np.random.rand(self.dim) * (neighborhood_best_position - particle)

                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_positions[i], personal_best_scores[i] = particles[i], score
                    if score < global_best_score:
                        global_best_position = particles[i] + np.random.uniform(-0.01, 0.01, self.dim)
                        global_best_score = score

            self.temperature *= self.cooling_rate
            self.social_constant = max(0.8, self.social_constant * 0.98)
            self.inertia_weight = max(0.4, self.inertia_weight * 0.99)

        return global_best_position
    
    def find_neighborhood_best(self, particles, scores, index, radius):
        distances = np.linalg.norm(particles - particles[index], axis=1)
        neighbors = np.where((distances < radius) & (distances > 0))[0]
        if len(neighbors) > 0:
            best_neighbor_idx = neighbors[np.argmin(scores[neighbors])]
            return particles[best_neighbor_idx]
        else:
            return particles[index]