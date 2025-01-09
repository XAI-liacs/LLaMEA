import numpy as np

class QiEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.qubits = np.random.uniform(-np.pi, np.pi, (self.population_size, self.dim))
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7
        self.evals = 0

    def _quantum_to_classical(self, qubits):
        """Convert qubits to classical representation using probability amplitudes."""
        positions = np.sin(qubits) ** 2
        return positions

    def _evaluate_population(self, func, positions):
        """Evaluate the population by converting the positions."""
        scores = np.array([func(positions[i]) for i in range(self.population_size)])
        self.evals += self.population_size
        return scores

    def _selection(self, positions, scores):
        """Select parents based on their scores using tournament selection."""
        selected = []
        for _ in range(self.population_size):
            i, j = np.random.choice(self.population_size, 2, replace=False)
            if scores[i] < scores[j]:
                selected.append(positions[i])
            else:
                selected.append(positions[j])
        return np.array(selected)

    def _crossover(self, parent1, parent2):
        """Perform crossover with a defined probability."""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim - 1)
            child = np.concatenate([parent1[:point], parent2[point:]])
            return child
        return parent1 if np.random.rand() < 0.5 else parent2

    def _mutate(self, qubit):
        """Mutate qubit angles with a defined probability."""
        if np.random.rand() < self.mutation_rate:
            qubit += np.random.uniform(-0.1, 0.1, size=self.dim)
        return qubit

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        while self.evals < self.budget:
            positions = np.clip(self._quantum_to_classical(self.qubits) * (ub - lb) + lb, lb, ub)
            scores = self._evaluate_population(func, positions)
            parents = self._selection(self.qubits, scores)
            
            new_qubits = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1)
                new_qubits.extend([self._mutate(child1), self._mutate(child2)])
            
            self.qubits = np.array(new_qubits)

            if self.evals >= self.budget:
                break

        best_index = np.argmin(scores)
        best_position = positions[best_index]
        best_score = scores[best_index]

        return best_position, best_score