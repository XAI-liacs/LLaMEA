import numpy as np

class AQIFA:
    def __init__(self, budget, dim, num_fireflies=20, absorption=1.0, attractiveness=0.5, beta_min=0.2, quantum_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.num_fireflies = num_fireflies
        self.absorption = absorption
        self.attractiveness = attractiveness
        self.beta_min = beta_min
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        fireflies = self.initialize_fireflies(lb, ub)
        intensities = np.array([func(firefly) for firefly in fireflies])
        self.evaluations += self.num_fireflies
        
        while self.evaluations < self.budget:
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if intensities[i] > intensities[j]:
                        distance = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta_min + (self.attractiveness - self.beta_min) * np.exp(-self.absorption * distance**2)
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        
                        if np.random.rand() < self.quantum_prob:
                            fireflies[i] = self.quantum_perturbation(fireflies[i], lb, ub)
                        
                        new_intensity = func(fireflies[i])
                        self.evaluations += 1
                        
                        if new_intensity < intensities[i]:
                            intensities[i] = new_intensity

                        if self.evaluations >= self.budget:
                            break

        best_index = np.argmin(intensities)
        return fireflies[best_index]

    def initialize_fireflies(self, lb, ub):
        return np.random.uniform(lb, ub, (self.num_fireflies, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)