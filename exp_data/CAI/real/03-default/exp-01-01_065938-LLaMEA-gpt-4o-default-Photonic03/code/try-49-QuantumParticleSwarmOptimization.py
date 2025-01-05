import numpy as np

class QuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7   # inertia weight
        self.quantum_radius = 0.1
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(x) for x in personal_best])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                # Update velocity using cognitive and social components
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - population[i]) +
                                 self.c2 * r2 * (global_best - population[i]))
                
                # Quantum behavior: add randomness to explore
                quantum_bounce = np.random.uniform(-self.quantum_radius, self.quantum_radius, self.dim)
                velocities[i] += quantum_bounce

                # Update position
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], lb, ub)

                # Evaluate fitness
                fitness = func(population[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness

                    # Update global best
                    if fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best = population[i]

            # Adaptive inertia weight
            self.w = 0.9 - (0.5 * evaluations / self.budget)

            self.history.append(global_best)

        return global_best