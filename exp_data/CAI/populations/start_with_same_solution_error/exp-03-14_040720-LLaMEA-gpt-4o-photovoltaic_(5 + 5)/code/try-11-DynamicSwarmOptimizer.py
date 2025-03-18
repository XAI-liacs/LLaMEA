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
        inertia_weight = 0.9
        c1, c2 = 1.5, 1.5

        while evaluations < self.budget:
            # Nonlinear decay for inertia weight
            inertia_weight = 0.5 + 0.4 * (0.5 * (1 + np.cos(np.pi * evaluations / self.budget)))  # Changed this line
            c1 = 1.5 + 0.5 * (evaluations / self.budget)  # Changed this line
            c2 = 1.5 - 0.5 * (evaluations / self.budget)  # Changed this line
            
            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 c2 * r2 * (global_best_position - swarm[i]))
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
            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                F = 0.5 + 0.3 * (1 - evaluations / self.budget)  # Changed this line
                mutant = np.clip(a + F * (b - c), lb, ub)  # Changed this line
                cr = 0.9 - 0.2 * (evaluations / self.budget)  # Changed this line
                trial = np.where(np.random.rand(self.dim) < cr, mutant, swarm[i])
                
                fitness = func(trial)
                evaluations += 1
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = np.copy(trial)
                    
                    if fitness < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = np.copy(personal_best_positions[i])

        return global_best_position, personal_best_scores[global_best_index]