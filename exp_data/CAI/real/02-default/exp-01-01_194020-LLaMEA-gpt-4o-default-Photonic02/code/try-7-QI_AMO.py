import numpy as np

class QI_AMO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.memory_archive = []
        self.archive_size = 20
        self.population_size = 10

    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def update_memory_archive(self, solution, value):
        self.memory_archive.append((solution.copy(), value))
        self.memory_archive = sorted(self.memory_archive, key=lambda x: x[1])[:self.archive_size]

    def quantum_superposition(self, lb, ub):
        if self.memory_archive:
            weights = np.exp(-np.arange(len(self.memory_archive)))
            weights /= np.sum(weights)
            idx = np.random.choice(len(self.memory_archive), p=weights)
            selected_solution, _ = self.memory_archive[idx]
            quantum_perturbation = np.random.uniform(-0.5, 0.5, self.dim)
            candidate = selected_solution + quantum_perturbation * (ub - lb)
            return np.clip(candidate, lb, ub)
        else:
            return lb + (ub - lb) * np.random.rand(self.dim)

    def local_refinement(self, solution, lb, ub):
        perturbation = np.random.normal(0, 0.01, self.dim)
        new_solution = solution + perturbation * (ub - lb)
        return np.clip(new_solution, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            candidate_solutions = []
            for _ in range(self.population_size):
                if evaluations < self.budget // 2:
                    candidate = self.quantum_superposition(lb, ub)
                else:
                    candidate = self.local_refinement(self.best_solution, lb, ub) if self.best_solution is not None else self.quantum_superposition(lb, ub)
                candidate_solutions.append(candidate)

            for solution in candidate_solutions:
                value = func(solution)
                evaluations += 1
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = solution
                self.update_memory_archive(solution, value)

                if evaluations >= self.budget:
                    break

        return self.best_solution, self.best_value