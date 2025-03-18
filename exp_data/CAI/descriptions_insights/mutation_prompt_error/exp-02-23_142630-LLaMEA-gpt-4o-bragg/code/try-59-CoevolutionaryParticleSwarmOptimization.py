import numpy as np

class CoevolutionaryParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.history = []
        self.velocity_clamp = 0.1

    def adapt_weights(self, diversity, eval_ratio):
        self.inertia_weight = 0.7 + 0.3 * (1 - diversity)
        self.cognitive_weight = 1.5 * (1 + eval_ratio)
        self.social_weight = 1.5 * (1 - eval_ratio)

    def update_velocity(self, particle_velocity, particle_position, personal_best, global_best):
        cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (personal_best - particle_position)
        social_component = self.social_weight * np.random.rand(self.dim) * (global_best - particle_position)
        new_velocity = self.inertia_weight * particle_velocity + cognitive_component + social_component
        return np.clip(new_velocity, -self.velocity_clamp, self.velocity_clamp)

    def __call__(self, func):
        self.bounds = func.bounds
        particles = np.random.uniform(self.bounds.lb, self.bounds.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.swarm_size, self.dim))
        personal_bests = particles.copy()
        personal_best_scores = np.array([func(p) for p in personal_bests])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_bests[global_best_idx]

        self.history.extend(personal_best_scores)

        evaluations = self.swarm_size
        while evaluations < self.budget:
            diversity = np.std(personal_best_scores) / (np.mean(personal_best_scores) + 1e-9)
            eval_ratio = evaluations / self.budget
            self.adapt_weights(diversity, eval_ratio)

            for i in range(self.swarm_size):
                velocities[i] = self.update_velocity(velocities[i], particles[i], personal_bests[i], global_best)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.bounds.lb, self.bounds.ub)

                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_bests[i] = particles[i]
                    personal_best_scores[i] = score

                    if score < personal_best_scores[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_bests[i]

            self.history.extend(personal_best_scores)

        return global_best, personal_best_scores[global_best_idx], self.history