import numpy as np

class QISE:
    def __init__(self, budget, dim, num_agents=30, alpha=0.75, beta=0.25):
        self.budget = budget
        self.dim = dim
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta = beta
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        positions = self.initialize_agents(lb, ub)
        velocities = np.random.uniform(-1, 1, (self.num_agents, self.dim))
        
        while self.evaluations < self.budget:
            for i in range(self.num_agents):
                personal_best_position = positions[i]
                personal_best_value = func(personal_best_position)
                self.evaluations += 1

                # Quantum-inspired update
                quantum_position = self.quantum_update(positions[i], best_global_position, lb, ub)
                quantum_value = func(quantum_position)
                self.evaluations += 1

                if quantum_value < personal_best_value:
                    personal_best_position = quantum_position
                    personal_best_value = quantum_value
                
                if personal_best_value < best_global_value:
                    best_global_position = personal_best_position
                    best_global_value = personal_best_value

                if self.evaluations >= self.budget:
                    break

            # Perform dynamic evolutionary operations
            self.evolve_agents(positions, lb, ub)
            
            if self.evaluations >= self.budget:
                break

        return best_global_position

    def initialize_agents(self, lb, ub):
        return np.random.uniform(lb, ub, (self.num_agents, self.dim))

    def quantum_update(self, position, global_best, lb, ub):
        if global_best is None:
            direction = np.random.uniform(-1, 1, self.dim)
        else:
            direction = global_best - position

        quantum_jump = self.alpha * direction + self.beta * np.random.uniform(-1, 1, self.dim) * (ub - lb)
        new_position = np.clip(position + quantum_jump, lb, ub)
        return new_position

    def evolve_agents(self, agents, lb, ub):
        for i in range(self.num_agents):
            if np.random.rand() < 0.1:  # Mutation rate
                mutation = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
                agents[i] = np.clip(agents[i] + mutation, lb, ub)
        
        for i in range(0, self.num_agents, 2):
            if i+1 < self.num_agents and np.random.rand() < 0.2:  # Crossover rate
                crossover_point = np.random.randint(1, self.dim)
                agents[i][:crossover_point], agents[i+1][:crossover_point] = (
                    agents[i+1][:crossover_point].copy(), agents[i][:crossover_point].copy())