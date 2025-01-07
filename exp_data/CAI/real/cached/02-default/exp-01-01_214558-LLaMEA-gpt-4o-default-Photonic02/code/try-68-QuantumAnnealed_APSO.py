import numpy as np

class QuantumAnnealed_APSO:
    def __init__(self, budget, dim, base_group_size=10, inertia=0.5, cognitive=1.5, social=1.5, quantum_prob=0.2, initial_temp=100, cooling_rate=0.99):
        self.budget = budget
        self.dim = dim
        self.base_group_size = base_group_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.quantum_prob = quantum_prob
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')
        
        particles = self.initialize_particles(self.base_group_size, lb, ub)
        velocities = self.initialize_velocities(self.base_group_size)
        temperatures = np.full(self.base_group_size, self.initial_temp)

        while self.evaluations < self.budget:
            for i in range(self.base_group_size):
                position = particles[i]

                if np.random.rand() < self.quantum_prob:
                    position = self.quantum_perturbation(position, lb, ub)

                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.random(self.dim) * np.subtract(particles[i], position) +
                                 self.social * np.random.random(self.dim) * (best_global_position - position if best_global_position is not None else 0))
                
                new_position = np.clip(position + velocities[i], lb, ub)

                if self.acceptance_probability(func(position), func(new_position), temperatures[i]) > np.random.rand():
                    position = new_position
                    particles[i] = position

                value = func(position)
                self.evaluations += 1

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = position

                temperatures[i] *= self.cooling_rate

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_particles(self, group_size, lb, ub):
        return np.random.uniform(lb, ub, (group_size, self.dim))

    def initialize_velocities(self, group_size):
        return np.random.uniform(-1, 1, (group_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / temperature)