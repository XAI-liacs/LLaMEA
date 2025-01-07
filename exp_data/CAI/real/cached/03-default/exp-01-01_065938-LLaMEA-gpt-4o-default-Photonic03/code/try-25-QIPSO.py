import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.w = 0.7   # Inertia weight
        self.global_best = None
        self.global_best_fitness = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros_like(positions)
        personal_best = positions.copy()
        personal_best_fitness = np.array([func(x) for x in personal_best])
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - positions[i]) +
                                 self.c2 * r2 * (self.global_best - positions[i] if self.global_best is not None else np.zeros(self.dim)))

                # Quantum-inspired exploration: apply random phase shifts
                quantum_shift = np.random.uniform(-np.pi, np.pi, self.dim)
                quantum_positions = positions[i] + np.sin(quantum_shift) * (ub - lb)
                quantum_positions = np.clip(quantum_positions, lb, ub)

                # Evaluate both classical and quantum positions
                trial_fitness = func(quantum_positions)
                evaluations += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = quantum_positions
                    personal_best_fitness[i] = trial_fitness

                if trial_fitness < self.global_best_fitness:
                    self.global_best = quantum_positions
                    self.global_best_fitness = trial_fitness

                # Update position
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

        return self.global_best