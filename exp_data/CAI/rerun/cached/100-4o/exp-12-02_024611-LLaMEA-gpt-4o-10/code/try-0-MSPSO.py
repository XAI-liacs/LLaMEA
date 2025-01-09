import numpy as np

class MSPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [-5.0, 5.0]
        self.num_particles = 30
        self.num_swarms = 5
        self.velocity_max = 0.2 * (self.bounds[1] - self.bounds[0])
        self.c1 = 1.5  # personal best weight
        self.c2 = 1.5  # global best weight

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.velocity_max, self.velocity_max, (self.num_particles, self.dim))
        personal_bests = particles.copy()
        personal_best_scores = np.array([func(x) for x in personal_bests])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_bests[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evals = self.num_particles
        while evals < self.budget:
            for swarm_id in range(self.num_swarms):
                swarm_start = swarm_id * self.num_particles // self.num_swarms
                swarm_end = (swarm_id + 1) * self.num_particles // self.num_swarms
                for i in range(swarm_start, swarm_end):
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (velocities[i] 
                                     + self.c1 * r1 * (personal_bests[i] - particles[i])
                                     + self.c2 * r2 * (global_best - particles[i]))
                    velocities[i] = np.clip(velocities[i], -self.velocity_max, self.velocity_max)
                    particles[i] += velocities[i]
                    particles[i] = np.clip(particles[i], self.bounds[0], self.bounds[1])

                    score = func(particles[i])
                    evals += 1
                    if score < personal_best_scores[i]:
                        personal_bests[i] = particles[i]
                        personal_best_scores[i] = score
                        if score < global_best_score:
                            global_best = particles[i]
                            global_best_score = score

                    if evals >= self.budget:
                        break
                if evals >= self.budget:
                    break
        return global_best