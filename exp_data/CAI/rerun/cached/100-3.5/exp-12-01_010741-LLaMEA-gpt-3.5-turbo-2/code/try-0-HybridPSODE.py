import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size

    def __call__(self, func):
        def pso_update(particles, g_best, w=0.5, c1=1.5, c2=1.5):
            for i in range(len(particles)):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                v_i = w * particles[i]['velocity'] + c1 * r1 * (particles[i]['p_best_pos'] - particles[i]['position']) + c2 * r2 * (g_best['position'] - particles[i]['position'])
                particles[i]['position'] += v_i
                particles[i]['velocity'] = v_i
                particles[i]['position'] = np.clip(particles[i]['position'], -5.0, 5.0)
            return particles

        def de_update(population, f=0.5, cr=0.9):
            for i in range(len(population)):
                idxs = [idx for idx in range(len(population)) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector = population[a]['position'] + f * (population[b]['position'] - population[c]['position'])
                cross_points = np.random.rand(self.dim) < cr
                trial_vector = np.where(cross_points, mutant_vector, population[i]['position'])
                trial_vector = np.clip(trial_vector, -5.0, 5.0)
                if func(trial_vector) < population[i]['fitness']:
                    population[i]['position'] = trial_vector
                    population[i]['fitness'] = func(trial_vector)
            return population

        particles = [{'position': np.random.uniform(-5.0, 5.0, self.dim), 'velocity': np.zeros(self.dim), 'fitness': np.inf, 'p_best_pos': np.zeros(self.dim), 'p_best_fit': np.inf} for _ in range(self.population_size)]
        g_best = {'position': np.zeros(self.dim), 'fitness': np.inf}

        for _ in range(self.max_iter):
            for i in range(len(particles)):
                fitness_i = func(particles[i]['position'])
                if fitness_i < particles[i]['fitness']:
                    particles[i]['p_best_pos'] = particles[i]['position'].copy()
                    particles[i]['p_best_fit'] = fitness_i
                    if fitness_i < g_best['fitness']:
                        g_best['position'] = particles[i]['position'].copy()
                        g_best['fitness'] = fitness_i

            particles = pso_update(particles, g_best)
            particles = de_update(particles)

        return g_best['position']