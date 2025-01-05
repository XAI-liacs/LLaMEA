import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive constant
        self.c2 = 1.5  # Social constant
        self.q_factor = 0.2  # Quantum factor to introduce randomness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in particles])
        personal_best = particles.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = particles[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_component = self.c1 * r1 * (personal_best[i] - particles[i])
                social_component = self.c2 * r2 * (global_best - particles[i])
                quantum_noise = self.q_factor * np.random.normal(0, 1, self.dim)

                velocity[i] = (self.w * velocity[i] + cognitive_component + social_component + quantum_noise)
                particles[i] = np.clip(particles[i] + velocity[i], lb, ub)

                current_fitness = func(particles[i])
                evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best = particles[i]

        return global_best