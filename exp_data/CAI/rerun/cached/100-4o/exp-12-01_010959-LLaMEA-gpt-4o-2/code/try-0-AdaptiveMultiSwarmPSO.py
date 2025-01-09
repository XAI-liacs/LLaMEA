import numpy as np

class AdaptiveMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_swarms = 5  # Number of swarms
        self.particles_per_swarm = 10  # Particles per swarm
        self.c1 = 2.05  # Cognitive component
        self.c2 = 2.05  # Social component
        self.w = 0.729  # Inertia weight
        self.local_search_prob = 0.3  # Probability to perform local search

    def __call__(self, func):
        num_particles = self.num_swarms * self.particles_per_swarm
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(num_particles, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            scores = np.array([func(pos) for pos in positions])
            evaluations += num_particles

            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]

            # Update global best
            min_index = np.argmin(personal_best_scores)
            if personal_best_scores[min_index] < global_best_score:
                global_best_score = personal_best_scores[min_index]
                global_best_position = personal_best_positions[min_index]

            # Adaptive swarm behavior
            for swarm_idx in range(self.num_swarms):
                start_idx = swarm_idx * self.particles_per_swarm
                end_idx = start_idx + self.particles_per_swarm
                swarm_positions = positions[start_idx:end_idx]
                swarm_velocities = velocities[start_idx:end_idx]
                swarm_personal_bests = personal_best_positions[start_idx:end_idx]

                # Update velocities and positions
                r1, r2 = np.random.rand(2, self.particles_per_swarm, self.dim)
                swarm_velocities = (self.w * swarm_velocities + 
                                    self.c1 * r1 * (swarm_personal_bests - swarm_positions) +
                                    self.c2 * r2 * (global_best_position - swarm_positions))
                swarm_positions += swarm_velocities

                # Apply bounds
                np.clip(swarm_positions, self.lower_bound, self.upper_bound, out=swarm_positions)

                # Randomized Local Search
                if np.random.rand() < self.local_search_prob:
                    local_search_idx = np.random.choice(range(self.particles_per_swarm))
                    candidate_position = swarm_positions[local_search_idx] + np.random.uniform(-0.1, 0.1, self.dim)
                    candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                    candidate_score = func(candidate_position)
                    evaluations += 1

                    if candidate_score < personal_best_scores[start_idx + local_search_idx]:
                        personal_best_scores[start_idx + local_search_idx] = candidate_score
                        personal_best_positions[start_idx + local_search_idx] = candidate_position
                        if candidate_score < global_best_score:
                            global_best_score = candidate_score
                            global_best_position = candidate_position

                # Update swarm positions and velocities
                positions[start_idx:end_idx] = swarm_positions
                velocities[start_idx:end_idx] = swarm_velocities

        return global_best_position, global_best_score