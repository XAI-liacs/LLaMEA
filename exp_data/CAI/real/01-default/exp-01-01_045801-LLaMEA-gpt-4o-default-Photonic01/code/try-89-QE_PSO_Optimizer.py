import numpy as np

class QE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.omega = 0.5  # inertia weight
        self.phi_p = 0.5  # cognitive component
        self.phi_g = 0.5  # social component
        self.population = None
        self.velocities = None
        self.best_positions = None
        self.best_personal_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        self.best_positions = self.population.copy()
        self.best_personal_scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_global_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_global_best(self):
        min_index = np.argmin(self.best_personal_scores)
        if self.best_personal_scores[min_index] < self.global_best_score:
            self.global_best_score = self.best_personal_scores[min_index]
            self.global_best_position = self.best_positions[min_index].copy()

    def adapt_parameters(self):
        self.omega = 0.5 + np.random.rand() / 2.0
        self.phi_p = 0.5 + np.random.rand() / 2.0
        self.phi_g = 0.5 + np.random.rand() / 2.0

    def quantum_tunneling(self, position, lb, ub):
        tunneling_vector = np.random.uniform(-1, 1, self.dim)
        quantum_position = position + tunneling_vector * np.random.uniform(0, np.abs(position - self.global_best_position))
        return np.clip(quantum_position, lb, ub)

    def update_particles(self, lb, ub):
        self.adapt_parameters()
        for i in range(self.population_size):
            r_p, r_g = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocities[i] = (self.omega * self.velocities[i] +
                                  self.phi_p * r_p * (self.best_positions[i] - self.population[i]) +
                                  self.phi_g * r_g * (self.global_best_position - self.population[i]))
            self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

            quantum_position = self.quantum_tunneling(self.population[i], lb, ub)
            quantum_score = self.evaluate(quantum_position)
            
            current_score = self.evaluate(self.population[i])
            self.evaluations += 2

            if quantum_score < current_score:
                self.population[i] = quantum_position
                current_score = quantum_score

            if current_score < self.best_personal_scores[i]:
                self.best_personal_scores[i] = current_score
                self.best_positions[i] = self.population[i].copy()

        self.update_global_best()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.update_particles(lb, ub)
            if self.evaluations >= self.budget:
                break

        return {'solution': self.global_best_position, 'fitness': self.global_best_score}