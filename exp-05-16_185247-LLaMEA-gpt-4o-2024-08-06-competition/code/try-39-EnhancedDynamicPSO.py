import numpy as np

class EnhancedDynamicPSO:
    def __init__(self, budget=10000, dim=10, n_particles=30, n_swarms=3):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.n_swarms = n_swarms
        self.lower_bound = -100.0
        self.upper_bound = 100.0

    def __call__(self, func):
        # Initialize multi-swarm particle positions and velocities
        swarms = [np.random.uniform(self.lower_bound, self.upper_bound, (self.n_particles, self.dim)) for _ in range(self.n_swarms)]
        velocities = [np.zeros((self.n_particles, self.dim)) for _ in range(self.n_swarms)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_scores = [np.full(self.n_particles, np.inf) for _ in range(self.n_swarms)]
        
        # Initialize global and swarm bests
        global_best_position = None
        global_best_score = np.inf
        swarm_best_positions = [None] * self.n_swarms
        swarm_best_scores = [np.inf] * self.n_swarms
        
        evaluations = 0
        stagnation_counter = [0] * self.n_swarms
        while evaluations < self.budget:
            previous_global_best_score = global_best_score
            for s in range(self.n_swarms):
                for i in range(self.n_particles):
                    if evaluations >= self.budget:
                        break

                    # Evaluate the fitness of the current particle
                    fitness = func(swarms[s][i])
                    evaluations += 1

                    # Update personal best
                    if fitness < personal_best_scores[s][i]:
                        personal_best_scores[s][i] = fitness
                        personal_best_positions[s][i] = swarms[s][i]

                    # Update swarm best
                    if fitness < swarm_best_scores[s]:
                        swarm_best_scores[s] = fitness
                        swarm_best_positions[s] = swarms[s][i]

                    # Update global best
                    if fitness < global_best_score:
                        global_best_score = fitness
                        global_best_position = swarms[s][i]

            if global_best_score == previous_global_best_score:
                for s in range(self.n_swarms):
                    stagnation_counter[s] += 1
            else:
                stagnation_counter = [0] * self.n_swarms

            # Update velocities and positions with multi-swarm communication
            for s in range(self.n_swarms):
                inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
                for i in range(self.n_particles):
                    r1, r2, r3 = np.random.rand(3)
                    learning_factor_cognitive = 2.0 - 1.5 * (evaluations / self.budget)
                    learning_factor_social = 2.0 + 1.5 * (evaluations / self.budget)
                    velocities[s][i] = (inertia_weight * velocities[s][i] +
                                 learning_factor_cognitive * r1 * (personal_best_positions[s][i] - swarms[s][i]) +
                                 learning_factor_social * r2 * (swarm_best_positions[s] - swarms[s][i]) +
                                 learning_factor_social * r3 * (global_best_position - swarms[s][i]))
                    
                    # Adaptive mutation mechanism based on stagnation
                    mutation_probability = min(0.1 + stagnation_counter[s] * 0.05, 0.5)
                    if np.random.rand() < mutation_probability:
                        velocities[s][i] += np.random.normal(0, 1, self.dim)
                    
                    velocities[s][i] = np.clip(velocities[s][i], -20 * (1 - evaluations/self.budget), 20 * (1 - evaluations/self.budget))
                    swarms[s][i] += velocities[s][i]

                    swarms[s][i] = np.clip(swarms[s][i], self.lower_bound, self.upper_bound)
        
        return global_best_score, global_best_position