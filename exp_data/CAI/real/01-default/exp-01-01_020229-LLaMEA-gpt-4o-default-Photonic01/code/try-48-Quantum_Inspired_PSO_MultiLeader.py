import numpy as np

class Quantum_Inspired_PSO_MultiLeader:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_min = 0.4
        self.w_max = 0.9
        self.q_factor = 0.8
        self.gaussian_scale = 0.15
        self.leader_fraction = 0.2

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
                # Adaptive Inertia Weight
                w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
                
                r1, r2 = np.random.rand(2)
                velocity[i] = (w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                
                adaptive_gaussian_scale = self.gaussian_scale * (1 - evaluations / self.budget)
                position[i] += (velocity[i] + 
                                self.q_factor * np.random.normal(scale=adaptive_gaussian_scale, size=self.dim))
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

            # Multi-Leader Approach
            leader_count = int(self.population_size * self.leader_fraction)
            sorted_indices = np.argsort(personal_best_value)
            leaders_positions = personal_best_position[sorted_indices[:leader_count]]
            leaders_values = personal_best_value[sorted_indices[:leader_count]]
            
            # Randomly re-initialize some particles using leaders' influence
            for j in range(leader_count):
                random_index = np.random.randint(self.population_size)
                influence = np.random.choice(leaders_positions)
                position[random_index] = np.random.uniform(lb, ub, self.dim) * 0.5 + influence * 0.5
                current_value = func(position[random_index])
                evaluations += 1
                
                if current_value < personal_best_value[random_index]:
                    personal_best_position[random_index] = position[random_index]
                    personal_best_value[random_index] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[random_index]
                    global_best_value = current_value

        return global_best_position, global_best_value