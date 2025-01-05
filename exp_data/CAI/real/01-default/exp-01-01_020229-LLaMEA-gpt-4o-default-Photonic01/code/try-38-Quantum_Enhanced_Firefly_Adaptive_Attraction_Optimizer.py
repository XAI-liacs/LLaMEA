import numpy as np

class Quantum_Enhanced_Firefly_Adaptive_Attraction_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5  # Randomness parameter
        self.beta0 = 1.0  # Base attraction
        self.gamma = 1.0  # Light absorption coefficient
        self.quantum_scale = 0.1
        self.anomaly_threshold = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        light_intensity = np.array([func(p) for p in position])
        best_position = position[np.argmin(light_intensity)]
        best_value = np.min(light_intensity)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if light_intensity[j] < light_intensity[i]:
                        r = np.linalg.norm(position[i] - position[j])
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        position[i] += beta * (position[j] - position[i])
                        position[i] += self.alpha * (np.random.uniform(-0.5, 0.5, self.dim) +
                                                     self.quantum_scale * np.random.normal(size=self.dim))
                        position[i] = np.clip(position[i], lb, ub)
                
                current_value = func(position[i])
                evaluations += 1
                
                if current_value < light_intensity[i]:
                    light_intensity[i] = current_value

                if current_value < best_value:
                    best_position = position[i]
                    best_value = current_value

                if evaluations >= self.budget:
                    break

            # Anomaly Detection Mechanism: Random perturbation if stagnation is detected
            if np.std(light_intensity) < self.anomaly_threshold:
                perturb_index = np.random.randint(self.population_size)
                position[perturb_index] = np.random.uniform(lb, ub, self.dim)
                current_value = func(position[perturb_index])
                evaluations += 1
                
                if current_value < light_intensity[perturb_index]:
                    light_intensity[perturb_index] = current_value
                
                if current_value < best_value:
                    best_position = position[perturb_index]
                    best_value = current_value

        return best_position, best_value