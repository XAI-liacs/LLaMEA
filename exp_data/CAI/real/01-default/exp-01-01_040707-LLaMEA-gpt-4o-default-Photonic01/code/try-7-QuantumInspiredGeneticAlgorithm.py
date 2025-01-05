import numpy as np

class QuantumInspiredGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.q_states = None
        self.population = None
        self.best_solution = None
        self.best_score = float('inf')
        self.alpha = 0.3  # Rotation angle for Q-gates

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.q_states = np.pi * np.random.rand(self.population_size, self.dim)  # Quantum superposition states
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best_solution = None
        self.best_score = float('inf')

    def collapse_q_state(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        binary_values = np.cos(self.q_states) ** 2 > np.random.rand(self.population_size, self.dim)
        return lb + (ub - lb) * binary_values.astype(float)

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.population])
        best_idx = np.argmin(scores)
        if scores[best_idx] < self.best_score:
            self.best_score = scores[best_idx]
            self.best_solution = self.population[best_idx]
        return scores

    def quantum_rotation_gate(self, scores):
        for i in range(self.population_size):
            if scores[i] < self.best_score:
                self.q_states[i] = self.q_states[i] + self.alpha
            else:
                self.q_states[i] = self.q_states[i] - self.alpha

    def genetic_operations(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        new_population = np.copy(self.population)
        for i in range(0, self.population_size, 2):
            if i + 1 >= self.population_size:
                break
            parent1, parent2 = self.population[i], self.population[i+1]
            crossover_point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            new_population[i], new_population[i+1] = child1, child2

        mutation_prob = 1.0 / self.dim
        for i in range(self.population_size):
            if np.random.rand() < mutation_prob:
                mutation_index = np.random.randint(self.dim)
                new_population[i, mutation_index] = lb[mutation_index] + (ub[mutation_index] - lb[mutation_index]) * np.random.rand()
        
        self.population = np.clip(new_population, lb, ub)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            self.population = self.collapse_q_state(func.bounds)
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.quantum_rotation_gate(scores)
            self.genetic_operations(func.bounds)
        return self.best_solution, self.best_score