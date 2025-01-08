import numpy as np

class QuantumAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + int(2 * np.sqrt(self.dim))
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 1.7  
        self.social_coefficient = 1.3     
        self.global_best_position = None
        self.global_best_value = float('inf')

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Dynamically adjust population size
        population_size = self.initial_population_size * (1 + self.budget // 1000)  # Changed line
        
        particles = np.random.uniform(lb, ub, (population_size, self.dim))  # Changed line
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.full(population_size, float('inf'))
        
        evaluations = 0

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                fitness = func(particles[i])
                evaluations += 1

                if fitness < personal_best_values[i]:
                    personal_best_values[i] = fitness
                    personal_best_positions[i] = particles[i].copy()

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particles[i].copy()

            # Update velocities and positions
            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_term = self.cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i])
                self.cognitive_coefficient = 1.9 - 0.5 * (evaluations / self.budget)
                self.social_coefficient = 1.1 + 0.8 * (evaluations / self.budget)
                social_term = self.social_coefficient * r2 * (self.global_best_position - particles[i])  
                velocities[i] = self.inertia_weight * velocities[i] + cognitive_term + social_term
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

            # Improved adaptive mutation strategy
            velocity_magnitude = np.linalg.norm(velocities, axis=1)
            mutation_prob = 1.0 / (1.0 + np.exp(-0.2 * (velocity_magnitude / np.max(velocity_magnitude) - 0.5)))
            for i in range(population_size):
                if np.random.rand() < mutation_prob[i] * (1 - evaluations/self.budget):  # Changed line
                    direction = np.sign(self.global_best_position - particles[i])
                    mutation_strength = (personal_best_values[i] - self.global_best_value) / (np.abs(personal_best_values[i] + 1e-9) * 0.9)
                    mutation_vector = 0.15 * direction * np.random.uniform(0, 1, self.dim) * mutation_strength
                    particles[i] += mutation_vector
                    particles[i] = np.clip(particles[i], lb, ub)

            self.inertia_weight *= 0.97**(1 + evaluations/self.budget)

        return self.global_best_position