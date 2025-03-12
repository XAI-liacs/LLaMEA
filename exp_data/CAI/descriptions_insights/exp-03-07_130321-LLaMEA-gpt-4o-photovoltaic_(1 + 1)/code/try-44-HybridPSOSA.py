import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.8
        self.temperature = 1000
        self.cooling_rate = 0.96 
        self.min_inertia_weight = 0.4
        self.num_swarms = 5  # Introduce multiple swarms

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Dynamic population size based on budget and dimensionality
        self.population_size = int(max(5, self.budget / (10 * self.dim)))
        evaluations = 0
        swarms = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_scores = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        global_best_position = min((pos[np.argmin(scores)] for pos, scores in zip(personal_best_positions, personal_best_scores)), key=lambda x: func(x))
        global_best_score = func(global_best_position)
        evaluations += sum(len(scores) for scores in personal_best_scores)

        while evaluations < self.budget:
            for i, (swarm, velocity) in enumerate(zip(swarms, velocities)):
                r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
                # Adaptive velocity update
                cognitive_term = self.cognitive_coef * (r1 ** 2) * (personal_best_positions[i] - swarm)
                social_term = self.social_coef * (r2 ** 2) * (global_best_position - swarm)
                velocity = self.inertia_weight * velocity + cognitive_term + social_term
                swarm += 0.5 * velocity
                swarm = np.clip(swarm, lb, ub)
                scores = np.array([func(ind) for ind in swarm])
                evaluations += self.population_size

                better_mask = scores < personal_best_scores[i]
                personal_best_scores[i][better_mask] = scores[better_mask]
                personal_best_positions[i][better_mask] = swarm[better_mask]
                if min(scores) < global_best_score:
                    global_best_score = min(scores)
                    global_best_position = swarm[np.argmin(scores)]

            self.temperature *= self.cooling_rate
            self.inertia_weight = max(self.min_inertia_weight, self.inertia_weight * 0.99 + 0.01 * (global_best_score / min(map(np.mean, personal_best_scores))))

        return global_best_position