import numpy as np

class Improved_PSO_DE_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, p_c=0.8, f=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.p_c = p_c
        self.f = f

    def __call__(self, func):
        def fitness(x):
            return func(x)

        particles = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest = particles.copy()
        pbest_scores = np.array([fitness(p) for p in pbest])
        gbest = pbest[pbest_scores.argmin()].copy()
        gbest_score = fitness(gbest)

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                velocity_cognitive = 2.0 * r1 * (pbest[i] - particles[i])
                velocity_social = 2.0 * r2 * (gbest - particles[i])
                velocities[i] = 0.5 * velocities[i] + velocity_cognitive + velocity_social
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], -5.0, 5.0)

                if np.random.rand() < self.p_c:
                    chosen = np.random.choice(self.swarm_size, 3, replace=False)
                    mutant = particles[chosen]
                    v = particles[i] + self.f * (mutant[0] - mutant[1] + mutant[2])
                    v = np.clip(v, -5.0, 5.0)
                    v_score = fitness(v)

                    if v_score < pbest_scores[i]:
                        pbest[i], pbest_scores[i] = v, v_score

                        if v_score < gbest_score:
                            gbest, gbest_score = v.copy(), v_score

                    evaluations += 1
                    if evaluations >= self.budget:
                        break
                
        return gbest