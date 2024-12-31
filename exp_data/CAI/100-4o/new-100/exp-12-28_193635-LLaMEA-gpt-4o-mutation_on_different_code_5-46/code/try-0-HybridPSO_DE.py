import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10, swarm_size=20, mutation_factor=0.5, crossover_prob=0.7, inertia=0.7, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize swarm
        particles = np.random.uniform(low=lb, high=ub, size=(self.swarm_size, self.dim))
        velocities = np.random.uniform(low=-1, high=1, size=(self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.swarm_size, np.Inf)

        global_best_position = None
        global_best_score = np.Inf

        evaluations = 0

        while evaluations < self.budget:
            # Evaluate particles
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                score = func(particles[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            # Update particle velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

            # Differential Evolution step
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                candidates = np.random.choice(self.swarm_size, 3, replace=False)
                x0, x1, x2 = particles[candidates[0]], particles[candidates[1]], particles[candidates[2]]
                mutant = np.clip(x0 + self.mutation_factor * (x1 - x2), lb, ub)

                trial = np.copy(particles[i])
                for d in range(self.dim):
                    if np.random.rand() < self.crossover_prob:
                        trial[d] = mutant[d]

                trial_score = func(trial)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial

                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial

        self.f_opt = global_best_score
        self.x_opt = global_best_position

        return self.f_opt, self.x_opt