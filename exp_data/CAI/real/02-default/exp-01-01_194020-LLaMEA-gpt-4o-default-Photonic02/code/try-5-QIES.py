import numpy as np

class QIES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 10
        self.quantum_population = np.pi * np.random.rand(self.population_size, self.dim)

    def decode(self, quantum_state, lb, ub):
        binary_population = np.cos(quantum_state) ** 2
        solution = lb + (ub - lb) * binary_population
        return solution

    def quantum_rotation(self, quantum_state, direction_vector):
        return quantum_state + np.pi * direction_vector * np.random.rand(self.dim)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0

        while evaluations < self.budget:
            candidate_solutions = []
            for q_state in self.quantum_population:
                solution = self.decode(q_state, lb, ub)
                candidate_solutions.append(solution)
            
            for solution in candidate_solutions:
                value = func(solution)
                evaluations += 1
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = solution

                if evaluations >= self.budget:
                    break

            # Quantum-inspired rotation to explore new regions
            direction_vectors = np.random.choice([-1, 1], size=(self.population_size, self.dim))
            self.quantum_population = np.array([
                self.quantum_rotation(q, d) for q, d in zip(self.quantum_population, direction_vectors)
            ])

        return self.best_solution, self.best_value