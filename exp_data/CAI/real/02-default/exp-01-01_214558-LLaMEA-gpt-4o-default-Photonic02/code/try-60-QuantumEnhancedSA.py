import numpy as np
from collections import deque

class QuantumEnhancedSA:
    def __init__(self, budget, dim, initial_temp=100, cooling_rate=0.99, memory_size=5, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.memory_size = memory_size
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.tabu_list = deque(maxlen=self.memory_size)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_position = self.initialize_position(lb, ub)
        current_value = func(current_position)
        best_position = current_position
        best_value = current_value
        temperature = self.initial_temp
        
        while self.evaluations < self.budget:
            new_position = self.generate_neighbor(current_position, lb, ub)
            new_value = func(new_position)
            self.evaluations += 1
            self.tabu_list.append(tuple(new_position))

            if new_value < best_value or self.acceptance_probability(current_value, new_value, temperature) > np.random.rand():
                current_position = new_position
                current_value = new_value
                if new_value < best_value:
                    best_value = new_value
                    best_position = new_position

            temperature *= self.cooling_rate

            if self.evaluations >= self.budget:
                break
        
        return best_position

    def initialize_position(self, lb, ub):
        return np.random.uniform(lb, ub, self.dim)

    def generate_neighbor(self, position, lb, ub):
        if np.random.rand() < self.quantum_prob:
            return self.quantum_perturbation(position, lb, ub)
        return np.clip(position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1, lb, ub)

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def acceptance_probability(self, current_value, new_value, temperature):
        if new_value < current_value:
            return 1.0
        return np.exp((current_value - new_value) / max(temperature, 1e-10))