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
        c1, c2 = 1.5, 1.5

        while evaluations < self.budget:
            inertia_weight = 0.9 - 0.6 * (evaluations / self.budget)

            diversity = np.std(swarm, axis=0)
            dynamic_factor = np.mean(diversity)

            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 c2 * r2 * (global_best_position - swarm[i]))
                
                scale_factor = 1 / (1 + np.exp(-evaluations/self.budget))
                velocities[i] *= scale_factor
                velocities[i] = np.clip(velocities[i], -dynamic_factor, dynamic_factor)

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

            for i in range(population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                perturbation = np.random.normal(0, dynamic_factor, self.dim)  # Gaussian perturbation
                mutant = np.clip(a + 0.7 * (b - c) + perturbation, lb, ub)
                trial = np.where(np.random.rand(self.dim) < 0.9, mutant, swarm[i])
                
                fitness = func(trial)
                evaluations += 1
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = np.copy(trial)
                    
                    if fitness < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = np.copy(personal_best_positions[i])

        return global_best_position, personal_best_scores[global_best_index]