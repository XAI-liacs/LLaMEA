import numpy as np

class QuantumInspiredWaveFunctionOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + int(2 * np.sqrt(dim))
        self.min_population_size = 5
        self.max_velocity = 0.2
        self.alpha = 0.9  # Damping factor for interference pattern
        self.beta = 0.1   # Step size for wave interference
        self.dynamic_shrinkage = 0.95

    def wave_function(self, position, amplitude, frequency):
        return amplitude * np.sin(frequency * position)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        positions = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.zeros((population_size, self.dim))
        best_positions = positions.copy()
        best_scores = np.full(population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                score = func(positions[i])
                evaluations += 1
                if score < best_scores[i]:
                    best_scores[i] = score
                    best_positions[i] = positions[i].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

            amplitude = np.sqrt(global_best_score + 1e-9)
            frequency = self.beta * (evaluations / self.budget)

            for i in range(population_size):
                interference = self.wave_function(positions[i], amplitude, frequency)
                velocities[i] = self.alpha * velocities[i] + interference
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

            # Reduce population size over time to enhance exploitation
            population_size = max(self.min_population_size, int(self.initial_population_size * (self.dynamic_shrinkage ** (evaluations / self.budget))))
            positions = positions[:population_size]
            velocities = velocities[:population_size]
            best_positions = best_positions[:population_size]
            best_scores = best_scores[:population_size]

        return global_best_position, global_best_score