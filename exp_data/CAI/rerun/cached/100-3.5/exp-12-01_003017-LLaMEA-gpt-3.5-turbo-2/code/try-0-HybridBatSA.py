import numpy as np

class HybridBatSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.loudness = 0.5
        self.pulse_rate = 0.5
        self.alpha = 0.95
        self.gamma = 0.1
        self.temperature = 100.0

    def __call__(self, func):
        def levy_flight():
            beta = 1.5
            sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.abs(v) ** (1 / beta)
            return step

        def new_solution(current, best):
            new_pos = current + levy_flight()
            for i in range(self.dim):
                if np.random.rand() > self.pulse_rate:
                    new_pos[i] = best[i] + np.random.uniform(-1, 1) * (current[i] - best[i])
            return new_pos

        def acceptance_probability(current, new_pos):
            current_val = func(current)
            new_val = func(new_pos)
            if new_val < current_val or np.random.rand() < np.exp((current_val - new_val) / self.temperature):
                return True
            return False

        def annealing():
            return self.temperature * self.alpha

        solutions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best_solution = solutions[np.argmin([func(sol) for sol in solutions])]

        for _ in range(self.budget):
            for i in range(self.population_size):
                if np.random.rand() > self.loudness:
                    solutions[i] = new_solution(solutions[i], best_solution)
                if acceptance_probability(solutions[i], best_solution):
                    best_solution = solutions[i]

            self.temperature = annealing()

        return best_solution