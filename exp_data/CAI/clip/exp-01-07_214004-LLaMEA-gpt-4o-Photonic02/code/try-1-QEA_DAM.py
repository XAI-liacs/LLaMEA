import numpy as np

class QEA_DAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.theta_angles = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.init_population()
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                solution = self.decode_solution(self.theta_angles[i], lb, ub)
                score = func(solution)
                self.evaluations += 1
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = solution
            
            self.update_population()
            
        return self.best_solution, self.best_score

    def init_population(self):
        self.theta_angles = np.random.uniform(0, 2 * np.pi, (self.population_size, self.dim))

    def decode_solution(self, theta, lb, ub):
        binary_solution = np.sign(np.sin(theta))
        return lb + 0.5 * (binary_solution + 1) * (ub - lb)

    def update_population(self):
        for i in range(self.population_size):
            delta_theta = np.random.uniform(-0.1, 0.1, self.dim)
            self.theta_angles[i] += delta_theta
            self.theta_angles[i] = np.mod(self.theta_angles[i], 2 * np.pi)

            if np.random.rand() < 0.1:  # Dynamic angle modulation
                j = np.random.randint(0, self.population_size)
                self.theta_angles[i] = 0.5 * (self.theta_angles[i] + self.theta_angles[j])