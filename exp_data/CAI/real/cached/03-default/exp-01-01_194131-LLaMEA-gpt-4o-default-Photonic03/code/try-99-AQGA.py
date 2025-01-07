import numpy as np

class AQGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.quantum_mutation_rate = 0.05

    def quantum_mutation(self, parent):
        alpha = np.pi * np.random.rand(self.dim)
        beta = np.pi * np.random.rand(self.dim)
        quantum_bit = np.sin(alpha) * parent + np.cos(beta) * (1 - parent)
        return quantum_bit

    def tournament_selection(self):
        indices = np.random.choice(self.population_size, 3, replace=False)
        selected = indices[np.argmin(self.fitness[indices])]
        return selected

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def _evaluate(self, individual, func):
        individual = np.clip(individual, func.bounds.lb, func.bounds.ub)
        score = func(individual)
        self.evaluations += 1
        return score

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.fitness[i] = self._evaluate(self.positions[i], func)
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.positions[i]
            if self.evaluations >= self.budget:
                return self.best_solution

        while self.evaluations < self.budget:
            new_positions = []
            for _ in range(self.population_size):
                if np.random.rand() < self.quantum_mutation_rate:
                    parent = self.positions[self.tournament_selection()]
                    child = self.quantum_mutation(parent)
                else:
                    parent1 = self.positions[self.tournament_selection()]
                    parent2 = self.positions[self.tournament_selection()]
                    child = self.crossover(parent1, parent2)
                
                new_positions.append(child)

            for i in range(self.population_size):
                new_score = self._evaluate(new_positions[i], func)
                if new_score < self.fitness[i]:
                    self.positions[i] = new_positions[i]
                    self.fitness[i] = new_score

                if new_score < self.best_fitness:
                    self.best_fitness = new_score
                    self.best_solution = new_positions[i]

                if self.evaluations >= self.budget:
                    break

            if self.evaluations % (self.budget // 5) == 0:
                self.quantum_mutation_rate = min(0.2, self.quantum_mutation_rate + 0.01)

        return self.best_solution