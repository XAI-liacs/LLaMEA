import numpy as np

class Quantum_Enhanced_Adaptive_Differential_Swarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25
        self.cr = 0.9  # Crossover rate
        self.f = 0.8   # Differential weight
        self.inertia_weight_start = 0.9
        self.inertia_weight_end = 0.4
        self.q_factor = 0.1
        self.epsilon = 1e-8

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize the population
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            inertia_weight = self.inertia_weight_start - (
                (self.inertia_weight_start - self.inertia_weight_end) * 
                (evaluations / self.budget)
            )

            for i in range(self.population_size):
                # Differential mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = position[indices]
                mutant = a + self.f * (b - c)

                # Quantum-inspired crossover
                trial = np.copy(position[i])
                crossover_points = np.random.rand(self.dim) < self.cr
                trial[crossover_points] = mutant[crossover_points]

                trial_value = func(trial)
                evaluations += 1

                if trial_value < personal_best_value[i]:
                    personal_best_position[i] = trial
                    personal_best_value[i] = trial_value

                if trial_value < global_best_value:
                    global_best_position = trial
                    global_best_value = trial_value

                if evaluations >= self.budget:
                    break

                # Update velocity and position with adaptive inertia
                velocity[i] = (inertia_weight * velocity[i] +
                               self.q_factor * np.random.normal(size=self.dim))
                position[i] = position[i] + velocity[i]
                position[i] = np.clip(position[i], lb, ub)

        return global_best_position, global_best_value