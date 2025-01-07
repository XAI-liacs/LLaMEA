import numpy as np

class CoopBAT:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Number of bats
        self.f_min = 0            # Minimum frequency
        self.f_max = 2            # Maximum frequency
        self.alpha = 0.9          # Loudness reduction factor
        self.gamma = 0.9          # Pulse rate increase factor
        self.initial_loudness = 1.0
        self.initial_pulse_rate = 0.5
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize the population
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        
        loudness = np.full(self.population_size, self.initial_loudness)
        pulse_rate = np.full(self.population_size, self.initial_pulse_rate)

        fitness = np.array([func(p) for p in position])
        best_index = np.argmin(fitness)
        global_best_position = position[best_index]
        global_best_value = fitness[best_index]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                frequency = np.random.uniform(self.f_min, self.f_max)
                velocity[i] += (position[i] - global_best_position) * frequency
                candidate_position = position[i] + velocity[i]

                # Simple bounds check
                candidate_position = np.clip(candidate_position, lb, ub)

                # If a random number is greater than the pulse rate, perform a local search
                if np.random.rand() > pulse_rate[i]:
                    candidate_position = global_best_position + 0.001 * np.random.randn(self.dim)
                    candidate_position = np.clip(candidate_position, lb, ub)
                
                candidate_value = func(candidate_position)
                evaluations += 1

                # Check if the new solution is better and within the loudness criteria
                if (candidate_value < fitness[i]) and (np.random.rand() < loudness[i]):
                    position[i] = candidate_position
                    fitness[i] = candidate_value
                    loudness[i] *= self.alpha
                    pulse_rate[i] = self.initial_pulse_rate * (1 - np.exp(-self.gamma * evaluations/self.budget))

                    # Update global best found so far
                    if candidate_value < global_best_value:
                        global_best_position = candidate_position
                        global_best_value = candidate_value

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_value