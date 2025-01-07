import numpy as np

class Quantum_Enhanced_SOS:
    def __init__(self, budget, dim, population_size=10, quantum_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        population = self.initialize_population(lb, ub)
        partner_indices = np.random.randint(0, self.population_size, self.population_size)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                position = population[i]

                if np.random.rand() < self.quantum_prob:
                    position = self.quantum_perturbation(position, lb, ub)

                partner = population[partner_indices[i]]
                symbiotic_partner = self.symbiotic_interaction(position, partner, lb, ub)
                
                value = func(symbiotic_partner)
                self.evaluations += 1

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = symbiotic_partner

                population[i] = symbiotic_partner

                if self.evaluations >= self.budget:
                    break

            partner_indices = np.random.randint(0, self.population_size, self.population_size)

        return best_global_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
        return np.clip(q_position, lb, ub)

    def symbiotic_interaction(self, individual, partner, lb, ub):
        alpha = np.random.rand()
        new_position = alpha * individual + (1 - alpha) * partner
        return np.clip(new_position, lb, ub)