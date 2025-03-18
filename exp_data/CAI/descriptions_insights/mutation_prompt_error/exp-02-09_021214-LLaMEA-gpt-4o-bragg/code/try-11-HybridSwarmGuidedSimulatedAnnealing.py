import numpy as np

class HybridSwarmGuidedSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 10
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.temp_initial = 100.0
        self.temp_final = 1.0
        self.best_solution = None
        self.best_obj = float('inf')
    
    def initialize_particles(self, bounds):
        self.lb, self.ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_obj = np.array([float('inf')] * self.num_particles)
    
    def update_velocities_and_positions(self, temperature):
        for i in range(self.num_particles):
            cognitive_velocity = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.c2 * np.random.rand(self.dim) * (self.best_solution - self.positions[i])
            self.velocities[i] = 0.5 * self.velocities[i] + cognitive_velocity + social_velocity
            self.velocities[i] = np.clip(self.velocities[i], -1, 1)
            
            new_position = self.positions[i] + self.velocities[i]
            new_position = np.clip(new_position, self.lb, self.ub)
            new_obj = func(new_position)
            
            # Simulated annealing acceptance
            delta_obj = new_obj - self.personal_best_obj[i]
            if delta_obj < 0 or np.exp(-delta_obj / temperature) > np.random.rand():
                self.positions[i] = new_position
                self.personal_best_positions[i] = new_position
                self.personal_best_obj[i] = new_obj
                if new_obj < self.best_obj:
                    self.best_obj = new_obj
                    self.best_solution = new_position
    
    def __call__(self, func):
        self.initialize_particles(func.bounds)
        evaluations = 0
        temperature = self.temp_initial
        
        while evaluations < self.budget:
            objectives = [func(pos) for pos in self.positions]
            evaluations += self.num_particles
            
            for i in range(self.num_particles):
                if objectives[i] < self.personal_best_obj[i]:
                    self.personal_best_obj[i] = objectives[i]
                    self.personal_best_positions[i] = self.positions[i]
                if objectives[i] < self.best_obj:
                    self.best_obj = objectives[i]
                    self.best_solution = self.positions[i]
            
            temperature = self.temp_initial - (self.temp_initial - self.temp_final) * (evaluations / self.budget)
            self.update_velocities_and_positions(temperature)
        
        return self.best_solution