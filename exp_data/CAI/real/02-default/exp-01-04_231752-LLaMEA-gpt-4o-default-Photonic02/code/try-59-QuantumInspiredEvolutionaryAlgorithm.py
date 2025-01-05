import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.q_population = np.full((self.population_size, self.dim, 2), 1/np.sqrt(2))  # Initialize qubits
        self.best_solution = None
        self.best_score = np.inf
        self.alpha = 0.05  # Rotation angle

    def _measure_population(self, lb, ub):
        population = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for d in range(self.dim):
                if np.random.rand() < self.q_population[i, d, 0] ** 2:
                    population[i, d] = 1
        scaled_population = lb + (ub - lb) * population
        return scaled_population

    def _update_quantum_population(self, best_position):
        for i in range(self.population_size):
            for d in range(self.dim):
                theta = self.alpha * (2 * np.random.rand() - 1)
                if np.random.rand() < self.q_population[i, d, 0] ** 2:
                    desired_state = best_position[d]
                else:
                    desired_state = 1 - best_position[d]
                if desired_state == 1:
                    theta = -theta
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                q0, q1 = self.q_population[i, d]
                self.q_population[i, d, 0] = cos_theta * q0 - sin_theta * q1
                self.q_population[i, d, 1] = sin_theta * q0 + cos_theta * q1

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        eval_count = 0
        
        while eval_count < self.budget:
            population = self._measure_population(self.lb, self.ub)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                score = func(population[i])
                eval_count += 1
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = population[i]
            
            if self.best_solution is not None:
                best_binary = ((self.best_solution - self.lb) / (self.ub - self.lb)).round()
                self._update_quantum_population(best_binary)
        
        return self.best_solution, self.best_score