import numpy as np

class RefinedHybridSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 30  # Adjusted swarm size for efficiency
        self.alpha_cognitive = 0.3  # Increased cognitive influence
        self.alpha_social = 0.1  # Further reduced social influence
        self.inertia_start = 0.8  # Modified starting inertia
        self.inertia_end = 0.3  # Adjusted inertia reduction
        self.mutation_prob_start = 0.15  # Dynamic mutation with higher initial probability
        self.mutation_prob_end = 0.05  # Reduced mutation probability over time

    def __call__(self, func):
        particle_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        particle_velocities = np.random.uniform(-0.2, 0.2, (self.swarm_size, self.dim))  # Adjusted velocity range
        personal_best_positions = np.copy(particle_positions)
        personal_best_values = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            inertia_weight = self.inertia_end + (self.inertia_start - self.inertia_end) * ((self.budget - evaluations) / self.budget)
            mutation_prob = self.mutation_prob_end + (self.mutation_prob_start - self.mutation_prob_end) * ((self.budget - evaluations) / self.budget)
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                rand_cognitive = np.random.rand(self.dim)
                rand_social = np.random.rand(self.dim)
                particle_velocities[i] = (inertia_weight * particle_velocities[i]
                                         + self.alpha_cognitive * rand_cognitive * (personal_best_positions[i] - particle_positions[i])
                                         + self.alpha_social * rand_social * (global_best_position - particle_positions[i]))
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)
                
                if np.random.rand() < mutation_prob:
                    mutation_step = np.random.normal(0, 0.05, self.dim)  # Refined mutation step size
                    particle_positions[i] += mutation_step
                    particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)

                current_value = func(particle_positions[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = np.copy(particle_positions[i])

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = np.copy(particle_positions[i])

        return global_best_position, global_best_value