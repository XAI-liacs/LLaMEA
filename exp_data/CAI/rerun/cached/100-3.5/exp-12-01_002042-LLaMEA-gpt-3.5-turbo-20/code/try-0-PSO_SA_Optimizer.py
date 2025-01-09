import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim, num_particles=30, max_iter=1000, c1=2.0, c2=2.0, initial_temp=100.0, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        def sa_acceptance_probability(curr_cost, new_cost, temp):
            return np.exp((curr_cost - new_cost) / temp)

        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def evaluate_particle(particle):
            return func(particle)

        def pso_sa_optimize():
            particles = initialize_particles()
            global_best = particles[np.argmin([evaluate_particle(p) for p in particles])]
            global_best_cost = evaluate_particle(global_best)
            temp = self.initial_temp

            for _ in range(self.max_iter):
                for i, particle in enumerate(particles):
                    particle_cost = evaluate_particle(particle)
                    if particle_cost < evaluate_particle(global_best):
                        global_best = particle.copy()
                        global_best_cost = particle_cost

                    new_particle = particle + np.random.uniform(-1, 1, size=self.dim) * (global_best - particle) + np.random.uniform(-1, 1, size=self.dim) * (particles[np.random.randint(0, self.num_particles)] - particle)
                    new_particle_cost = evaluate_particle(new_particle)

                    if new_particle_cost < particle_cost or np.random.rand() < sa_acceptance_probability(particle_cost, new_particle_cost, temp):
                        particles[i] = new_particle

                temp *= self.cooling_rate

            return global_best

        return pso_sa_optimize()