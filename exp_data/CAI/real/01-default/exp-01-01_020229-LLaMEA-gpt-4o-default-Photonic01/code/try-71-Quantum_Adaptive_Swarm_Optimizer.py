import numpy as np

class Quantum_Adaptive_Swarm_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Slightly increased for better exploration
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.8  # Initial inertia weight
        self.q_factor = 0.9
        self.gaussian_scale = 0.05  # Reduced for precision in local search
        self.adaptive_rate = 0.995  # More gradual decrease of inertia weight
        self.temperature = 1.0
        self.elite_rate = 0.1  # Rate to trigger elite local search

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.w *= self.adaptive_rate
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                adaptive_gaussian_scale = self.gaussian_scale * (1 - evaluations / self.budget)
                annealing_factor = np.exp(-evaluations / (self.budget * self.temperature))
                
                position[i] += (velocity[i] + 
                                self.q_factor * np.random.normal(scale=adaptive_gaussian_scale, size=self.dim) +
                                annealing_factor * np.random.uniform(-1, 1, self.dim))
                position[i] = np.clip(position[i], lb, ub)

                current_value = func(position[i])
                evaluations += 1
                
                if current_value < personal_best_value[i]:
                    personal_best_position[i] = position[i]
                    personal_best_value[i] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if evaluations >= self.budget:
                    break

            # Elite Local Search for top performers
            elite_indices = np.argsort(personal_best_value)[:int(self.elite_rate * self.population_size)]
            for idx in elite_indices:
                perturbation = np.random.normal(0, adaptive_gaussian_scale, self.dim)
                elite_pos = personal_best_position[idx] + perturbation
                elite_pos = np.clip(elite_pos, lb, ub)
                elite_value = func(elite_pos)
                evaluations += 1

                if elite_value < personal_best_value[idx]:
                    personal_best_position[idx] = elite_pos
                    personal_best_value[idx] = elite_value
                
                if elite_value < global_best_value:
                    global_best_position = elite_pos
                    global_best_value = elite_value

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_value