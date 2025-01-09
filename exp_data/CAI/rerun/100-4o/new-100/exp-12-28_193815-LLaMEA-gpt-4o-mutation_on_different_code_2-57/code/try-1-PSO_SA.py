import numpy as np

class PSO_SA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.inertia_weight = 0.8  # Increased from 0.7 to 0.8
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.temperature = 100
        self.cooling_rate = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(individual) for individual in swarm])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                velocities[i] = (self.inertia_weight * velocities[i] + 
                                 self.cognitive_coefficient * np.random.rand() * (personal_best_positions[i] - swarm[i]) + 
                                 self.social_coefficient * np.random.rand() * (global_best_position - swarm[i]))
                
                swarm[i] = swarm[i] + velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                current_score = func(swarm[i])
                evaluations += 1

                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = swarm[i]

                    if current_score < global_best_score:
                        global_best_score = current_score
                        global_best_position = swarm[i]

                # Simulated Annealing acceptance criterion
                if np.random.rand() < np.exp((personal_best_scores[i] - current_score) / self.temperature):
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = swarm[i]
                    
                    if current_score < global_best_score:
                        global_best_score = current_score
                        global_best_position = swarm[i]

            self.temperature *= self.cooling_rate

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt