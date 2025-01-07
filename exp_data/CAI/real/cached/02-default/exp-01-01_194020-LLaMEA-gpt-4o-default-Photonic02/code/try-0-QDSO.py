import numpy as np

class QDSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 10
        self.photon_ratio = 0.1
        self.q_population = np.random.rand(self.population_size, dim)

    def quantum_update(self, lb, ub):
        r = np.random.rand(self.dim)
        binary_position = np.where(r < self.photon_ratio, 1, 0)
        new_position = (self.q_population + binary_position) % 2
        new_position = lb + (ub - lb) * new_position
        return new_position

    def local_search(self, solution, lb, ub):
        perturbation = np.random.uniform(-0.05, 0.05, self.dim)
        new_solution = solution + perturbation * (ub - lb)
        return np.clip(new_solution, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0

        while evaluations < self.budget:
            if evaluations % (self.budget // 2) < (self.budget // 4):
                # Quantum-inspired exploration phase
                candidate_solutions = [self.quantum_update(lb, ub) for _ in range(self.population_size)]
            else:
                # Local-exploitation phase
                candidate_solutions = [self.local_search(self.best_solution, lb, ub) for _ in range(self.population_size)]

            for solution in candidate_solutions:
                value = func(solution)
                evaluations += 1
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = solution

                if evaluations >= self.budget:
                    break

        return self.best_solution, self.best_value