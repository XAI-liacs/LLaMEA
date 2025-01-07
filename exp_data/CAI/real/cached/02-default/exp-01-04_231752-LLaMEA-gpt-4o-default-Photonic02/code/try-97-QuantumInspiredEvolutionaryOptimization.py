import numpy as np

class QuantumInspiredEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.gamma = 0.03  # Quantum interference factor
        self.elitism_rate = 0.2  # Rate for selecting elite individuals
        self.positions = None
        self.best_position = None
        self.best_score = np.inf

    def _initialize_population(self, lb, ub):
        self.positions = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.best_position = None
        self.best_score = np.inf

    def _quantum_interference(self, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < 0.5:
                interference = np.sin(self.gamma * np.pi * (ub - lb))
                self.positions[i] = self.positions[i] + interference * (np.random.rand(self.dim) - 0.5)
                self.positions[i] = np.clip(self.positions[i], lb, ub)

    def _select_elite(self, scores):
        elite_count = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(scores)[:elite_count]
        return self.positions[elite_indices]

    def _generate_offspring(self, elites, lb, ub):
        offspring_size = self.population_size - len(elites)
        offspring = np.empty((offspring_size, self.dim))
        for i in range(offspring_size):
            parent1, parent2 = np.random.choice(len(elites), 2, replace=False)
            crossover_point = np.random.randint(1, self.dim)
            offspring[i, :crossover_point] = elites[parent1, :crossover_point]
            offspring[i, crossover_point:] = elites[parent2, crossover_point:]
            mutation = (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.05
            offspring[i] = np.clip(offspring[i] + mutation, lb, ub)
        return offspring

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            scores = np.array([func(pos) for pos in self.positions])
            eval_count += self.population_size

            if np.min(scores) < self.best_score:
                self.best_score = np.min(scores)
                self.best_position = self.positions[np.argmin(scores)]

            elites = self._select_elite(scores)
            offspring = self._generate_offspring(elites, self.lb, self.ub)
            self.positions = np.vstack((elites, offspring))
            self._quantum_interference(self.lb, self.ub)

        return self.best_position, self.best_score