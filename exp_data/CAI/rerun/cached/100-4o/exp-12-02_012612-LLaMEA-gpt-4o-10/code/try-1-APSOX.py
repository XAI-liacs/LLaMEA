import numpy as np

class APSOX:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.swarm = None
        self.velocities = None
        self.best_personal_positions = None
        self.best_personal_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def initialize_swarm(self):
        self.swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_personal_positions = np.copy(self.swarm)
        self.best_personal_scores = np.full(self.population_size, np.inf)

    def update_particles(self, inertia, cognitive, social):
        r1 = np.random.random((self.population_size, self.dim))
        r2 = np.random.random((self.population_size, self.dim))
        self.velocities = (inertia * self.velocities +
                           cognitive * r1 * (self.best_personal_positions - self.swarm) +
                           social * r2 * (self.global_best_position - self.swarm))
        
        self.swarm += self.velocities
        np.clip(self.swarm, self.lower_bound, self.upper_bound, out=self.swarm)

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) > 0.5
        offspring = np.where(mask, parent1, parent2)
        np.clip(offspring, self.lower_bound, self.upper_bound, out=offspring)
        return offspring

    def __call__(self, func):
        self.initialize_swarm()
        inertia = 0.7
        cognitive = 1.4
        social = 1.4
        crossover_rate = 0.2

        for _ in range(self.budget):
            for i in range(self.population_size):
                score = func(self.swarm[i])
                self.evaluations += 1
                
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = np.copy(self.swarm[i])
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.swarm[i])
            
            self.update_particles(inertia, cognitive, social)
            
            if np.random.rand() < crossover_rate:
                indices = np.random.choice(self.population_size, 2, replace=False)
                offspring = self.crossover(self.swarm[indices[0]], self.swarm[indices[1]])
                offspring_score = func(offspring)
                self.evaluations += 1
                
                if offspring_score < self.global_best_score:
                    self.global_best_score = offspring_score
                    self.global_best_position = np.copy(offspring)

            if self.evaluations >= self.budget:
                break

        return self.global_best_position