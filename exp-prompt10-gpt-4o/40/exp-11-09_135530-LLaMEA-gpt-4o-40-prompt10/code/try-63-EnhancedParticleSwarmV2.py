import numpy as np

class EnhancedParticleSwarmV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.initial_population_size = 50
        self.min_population_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.3
        self.alpha = 0.5  # Crossover rate
        self.beta_initial = 0.8  # Differential mutation rate
        self.evaluations = 0
    
    def initialize_particles(self, size):
        particles = np.random.uniform(self.lb, self.ub, (size, self.dim))
        velocities = np.random.uniform(-1, 1, (size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores
    
    def adaptive_crossover(self, parent1, parent2):
        if np.random.rand() < self.alpha:
            cross_point = np.random.randint(1, self.dim)
            child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        else:
            child = (parent1 + parent2) / 2
        return np.clip(child, self.lb, self.ub)
    
    def differential_mutation(self, target, best, r1, r2, beta):
        mutated = target + beta * (best - target) + beta * (r1 - r2)
        return np.clip(mutated, self.lb, self.ub)
    
    def resize_population(self, population_size, personal_best_scores):
        if self.evaluations < self.budget * 0.5:
            return population_size
        else:
            return max(self.min_population_size, population_size // 2)

    def __call__(self, func):
        population_size = self.initial_population_size
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles(population_size)
        global_best_position = None
        global_best_score = np.inf
        
        while self.evaluations < self.budget:
            for i in range(population_size):
                score = func(particles[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)
            beta = self.beta_initial * (1 - self.evaluations / self.budget)
            
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                r1, r2 = np.random.choice(population_size, 2, replace=False)
                r1, r2 = personal_best_positions[r1], personal_best_positions[r2]
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.differential_mutation(particles[i], global_best_position, r1, r2, beta)
                particles[i] = np.clip(particles[i], self.lb, self.ub)
                
            if self.evaluations + population_size <= self.budget:
                for i in range(population_size):
                    parent1, parent2 = personal_best_positions[np.random.choice(population_size, 2, replace=False)]
                    child = self.adaptive_crossover(parent1, parent2)
                    score = func(child)
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = child
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = child
            
            population_size = self.resize_population(population_size, personal_best_scores)
            particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles(population_size)
        
        return global_best_position, global_best_score