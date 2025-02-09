import numpy as np

class QuantumInspiredAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + int(2 * np.sqrt(dim))
        self.min_population_size = 5
        self.alpha = 0.5  # Amplitude damping factor
        self.beta = 1.0   # Phase damping factor
        self.dynamic_shrinkage = 0.95

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        wave_functions = np.random.uniform(lb, ub, (population_size, self.dim, 2))
        personal_best_positions = wave_functions[:, :, 0].copy()
        personal_best_scores = np.full(population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            # Collapse wave functions to concrete positions
            positions = wave_functions[:, :, 0] + self.beta * (wave_functions[:, :, 1] - wave_functions[:, :, 0])
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

            # Update wave functions based on best known positions
            for i in range(population_size):
                r = np.random.rand(self.dim)
                wave_functions[i, :, 0] = self.alpha * wave_functions[i, :, 0] + r * (personal_best_positions[i] - wave_functions[i, :, 0])
                wave_functions[i, :, 1] = self.alpha * wave_functions[i, :, 1] + r * (global_best_position - wave_functions[i, :, 1])
                wave_functions[i] = np.clip(wave_functions[i], lb, ub)
            
            # Reduce population size over time to intensify search
            population_size = max(self.min_population_size, int(self.initial_population_size * (self.dynamic_shrinkage ** (evaluations / self.budget))))
            wave_functions = wave_functions[:population_size]
            personal_best_positions = personal_best_positions[:population_size]
            personal_best_scores = personal_best_scores[:population_size]

        return global_best_position, global_best_score