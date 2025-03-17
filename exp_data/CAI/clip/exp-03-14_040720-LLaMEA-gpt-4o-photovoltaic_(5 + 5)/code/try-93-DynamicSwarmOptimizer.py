import numpy as np

class DynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = max(20, 5 * self.dim)
        swarm = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(x) for x in swarm])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_index])

        evaluations = len(swarm)
        inertia_weight = 0.7

        while evaluations < self.budget:
            # Time-varying acceleration coefficients
            c1 = 2.5 - (2.5 - 0.5) * (evaluations / self.budget)
            c2 = 0.5 + (2.5 - 0.5) * (evaluations / self.budget)

            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 c2 * r2 * (global_best_position - swarm[i]))
                if np.random.rand() < 0.1:
                    velocities[i] += self._levy_flight(self.dim)  # Levy flight perturbation
                
                velocities[i] *= 0.9  # Adaptive velocity scaling
                swarm[i] = swarm[i] + velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)
                
                fitness = func(swarm[i])
                evaluations += 1
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = np.copy(swarm[i])
                    
                    if fitness < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = np.copy(personal_best_positions[i])
            
            if evaluations >= self.budget:
                break

            # Differential Evolution Mutation and Crossover
            elite_size = max(2, population_size // 10)  # Elite selection
            best_indices = np.argsort(personal_best_scores)[:elite_size]
            for i in range(population_size):
                indices = np.random.choice(best_indices, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                mutant = np.clip(a + 0.8 * (b - c), lb, ub)
                trial = np.where(np.random.rand(self.dim) < np.sin(np.pi * evaluations / self.budget), mutant, swarm[i])  # Modulated crossover rate
                
                fitness = func(trial)
                evaluations += 1
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = np.copy(trial)
                    
                    if fitness < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = np.copy(personal_best_positions[i])

        return global_best_position, personal_best_scores[global_best_index]
    
    def _levy_flight(self, dim, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / np.abs(v) ** (1 / beta)
        return step