import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.inertia = 0.5
        self.cognitive = 1.5
        self.social = 1.5
        self.temperature = 100.0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        positions = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        velocities = np.zeros((self.pop_size, self.dim))
        return positions, velocities

    def update_velocity(self, velocities, positions, pbest_positions, gbest_position):
        r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
        cognitive_component = self.cognitive * r1 * (pbest_positions - positions)
        social_component = self.social * r2 * (gbest_position - positions)
        new_velocities = self.inertia * velocities + cognitive_component + social_component
        return new_velocities

    def update_position(self, positions, velocities, bounds):
        new_positions = positions + velocities
        return np.clip(new_positions, bounds.lb, bounds.ub)

    def simulated_annealing(self, candidate, func, bounds):
        current_energy = func(candidate)
        for _ in range(100):  # Inner-loop of SA
            neighbor = candidate + np.random.uniform(-0.1, 0.1, self.dim)
            neighbor = np.clip(neighbor, bounds.lb, bounds.ub)
            neighbor_energy = func(neighbor)
            if neighbor_energy < current_energy or np.exp((current_energy - neighbor_energy) / self.temperature) > np.random.rand():
                candidate, current_energy = neighbor, neighbor_energy
            self.temperature *= 0.99  # Cooling schedule
        return candidate, current_energy

    def __call__(self, func):
        bounds = func.bounds
        positions, velocities = self.initialize_population(bounds)
        pbest_positions = positions.copy()
        pbest_values = np.array([func(p) for p in positions])
        gbest_position = pbest_positions[np.argmin(pbest_values)]
        gbest_value = np.min(pbest_values)

        evaluations = self.pop_size
        while evaluations < self.budget:
            velocities = self.update_velocity(velocities, positions, pbest_positions, gbest_position)
            positions = self.update_position(positions, velocities, bounds)

            for i in range(self.pop_size):
                curr_value = func(positions[i])
                evaluations += 1
                if curr_value < pbest_values[i]:
                    pbest_positions[i] = positions[i]
                    pbest_values[i] = curr_value
                if curr_value < gbest_value:
                    gbest_position = positions[i]
                    gbest_value = curr_value

                if evaluations >= self.budget:
                    break

            gbest_position, gbest_value = self.simulated_annealing(gbest_position, func, bounds)

        return gbest_position