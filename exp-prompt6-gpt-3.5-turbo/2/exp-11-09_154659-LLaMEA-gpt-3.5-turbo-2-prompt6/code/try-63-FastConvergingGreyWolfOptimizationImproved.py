import numpy as np

class FastConvergingGreyWolfOptimizationImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def get_alpha_beta_delta(wolves):
            sorted_wolves = sorted(wolves, key=lambda x: x['fitness'])
            return sorted_wolves[0], sorted_wolves[1], sorted_wolves[2]

        def update_position(wolf, alpha, beta, delta, prev_fitness, fitness_trend):
            fitness_improvement = prev_fitness - wolf['fitness']
            step_size = 1.0 - np.exp(-0.1 * fitness_improvement)  # Dynamic step size based on fitness improvement
            a = 1.8 - 1.6 * (np.arange(self.dim) / (self.dim - 1)) * step_size

            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = np.abs(C1 * alpha['position'] - wolf['position'])
            X1 = alpha['position'] - A1 * D_alpha

            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = np.abs(C2 * beta['position'] - wolf['position'])
            X2 = beta['position'] - A2 * D_beta

            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = np.abs(C3 * delta['position'] - wolf['position'])
            X3 = delta['position'] - A3 * D_delta

            new_position = (X1 + X2 + X3) / 3
            new_position = np.clip(new_position, -5.0, 5.0)

            if fitness_improvement > np.mean(fitness_trend):  # Adaptive mutation based on fitness trends
                new_position = new_position + np.random.normal(0, 0.1, self.dim)

            return new_position

        wolves = [{'position': np.random.uniform(-5.0, 5.0, self.dim),
                   'fitness': func(np.random.uniform(-5.0, 5.0, self.dim))} for _ in range(5)]
        fitness_values = [wolf['fitness'] for wolf in wolves]
        fitness_trend = []

        for i in range(2, self.budget - 3, 2):
            alpha, beta, delta = get_alpha_beta_delta(wolves)
            variance = np.var(fitness_values)
            if variance < 0.1:
                wolves.pop(-1)
            else:
                wolves.append({'position': np.random.uniform(-5.0, 5.0, self.dim),
                               'fitness': func(np.random.uniform(-5.0, 5.0, self.dim))})
                fitness_values.append(wolves[-1]['fitness'])
                fitness_trend.append(wolves[-1]['fitness'])

            for wolf in wolves:
                prev_fitness = wolf['fitness']
                wolf['position'] = update_position(wolf, alpha, beta, delta, prev_fitness, fitness_trend)
                wolf['fitness'] = func(wolf['position'])

        best_wolf = min(wolves, key=lambda x: x['fitness'])
        return best_wolf['position']