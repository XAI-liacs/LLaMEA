import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 30
        self.alpha = 0.05  # Mutation rate

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            q_state = np.random.rand(self.dim)  # Quantum bit state
            population.append({'position': position, 'q_state': q_state, 'best_value': float('inf')})
        return population

    def measure(self, q_state, lb, ub):
        theta = np.arccos(2 * q_state - 1)
        position = lb + (ub - lb) * ((1 + np.cos(theta))/2)
        return position

    def update_quantum_state(self, q_state, position, lb, ub):
        # Perturbate the quantum state
        new_q_state = np.clip(q_state + self.alpha * (np.random.rand(self.dim) - 0.5), 0, 1)
        new_position = self.measure(new_q_state, lb, ub)
        return new_q_state, new_position

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            for individual in population:
                individual['position'] = self.measure(individual['q_state'], lb, ub)
                value = func(individual['position'])
                evaluations += 1

                if value < individual['best_value']:
                    individual['best_value'] = value
                    individual['position'] = individual['position'].copy()

                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = individual['position'].copy()

                if evaluations >= self.budget:
                    break

            # Update quantum states
            for individual in population:
                individual['q_state'], individual['position'] = self.update_quantum_state(individual['q_state'], individual['position'], lb, ub)

        return self.best_solution, self.best_value