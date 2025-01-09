import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.positions = np.random.uniform(-5, 5, (self.swarm_size, self.dim))
        self.personal_best = np.copy(self.positions)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best = None
        self.global_best_value = np.inf
        self.func_evaluations = 0

    def differential_evolution(self, target_idx):
        indices = [i for i in range(self.swarm_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        mutant_vector = self.positions[a] + F * (self.positions[b] - self.positions[c])
        trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, self.positions[target_idx])
        trial_vector = np.clip(trial_vector, -5, 5)
        return trial_vector

    def __call__(self, func):
        inertia_weight = 0.7
        cognitive_weight = 1.5
        social_weight = 1.5

        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.func_evaluations >= self.budget:
                    break

                # Evaluate current position
                current_value = func(self.positions[i])
                self.func_evaluations += 1

                # Update personal best
                if current_value < self.personal_best_value[i]:
                    self.personal_best_value[i] = current_value
                    self.personal_best[i] = self.positions[i]

                # Update global best
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best = self.positions[i]

            for i in range(self.swarm_size):
                if self.func_evaluations >= self.budget:
                    break
                
                # Update velocity and position for PSO
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_weight * r1 * (self.personal_best[i] - self.positions[i]) +
                                    social_weight * r2 * (self.global_best - self.positions[i]))
                self.positions[i] += self.velocity[i]
                self.positions[i] = np.clip(self.positions[i], -5, 5)

                # Apply DE mutation
                trial_vector = self.differential_evolution(i)
                trial_value = func(trial_vector)
                self.func_evaluations += 1

                # Greedy selection
                if trial_value < current_value:
                    self.positions[i] = trial_vector

        return self.global_best, self.global_best_value