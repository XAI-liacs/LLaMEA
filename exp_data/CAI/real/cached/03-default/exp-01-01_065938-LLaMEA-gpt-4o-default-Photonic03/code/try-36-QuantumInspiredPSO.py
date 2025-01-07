import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.5   # Inertia weight
        self.alpha = 0.1  # Quantum parameter for enhanced exploration

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(personal_best_fitness)
        best_global_position = personal_best_positions[best_idx]
        best_global_fitness = personal_best_fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocities and positions using quantum inspired approach
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity_cognitive = self.c1 * r1 * (personal_best_positions[i] - pop[i])
                velocity_social = self.c2 * r2 * (best_global_position - pop[i])
                quantum_step = self.alpha * (np.random.rand(self.dim) * 2 - 1)
                velocities[i] = self.w * velocities[i] + velocity_cognitive + velocity_social + quantum_step
                
                # Update position
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], lb, ub)
                
                # Evaluate fitness
                current_fitness = func(pop[i])
                evaluations += 1

                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = pop[i]
                    personal_best_fitness[i] = current_fitness

                    # Update global best
                    if current_fitness < best_global_fitness:
                        best_global_position = pop[i]
                        best_global_fitness = current_fitness

            # Adaptive inertia weight for convergence refinement
            self.w = 0.4 + 0.5 * (1 - evaluations / self.budget)

        return best_global_position