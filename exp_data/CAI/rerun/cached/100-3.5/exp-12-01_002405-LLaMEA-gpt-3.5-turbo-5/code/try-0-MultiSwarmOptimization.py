import numpy as np

class MultiSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 5
        self.swarm_size = 10
        self.max_swarm_size = 20
        self.min_swarm_size = 5
        self.swarms = [np.random.uniform(-5.0, 5.0, (self.swarm_size, dim)) for _ in range(self.num_swarms)]
    
    def __call__(self, func):
        for _ in range(self.budget):
            for swarm in self.swarms:
                fitness = [func(p) for p in swarm]
                best_idx = np.argmin(fitness)
                best_particle = swarm[best_idx]

                for i, particle in enumerate(swarm):
                    if i != best_idx:
                        rand_swarm = np.random.choice(self.swarms)
                        rand_particle = rand_swarm[np.random.randint(0, len(rand_swarm))]
                        new_particle = particle + np.random.uniform(0, 1, self.dim) * (best_particle - particle) + np.random.uniform(0, 1, self.dim) * (rand_particle - particle)
                        if func(new_particle) < fitness[i]:
                            swarm[i] = new_particle
            
            # Dynamic swarm size adjustment
            avg_fitness = np.mean([func(p) for p in np.concatenate(self.swarms)])
            if avg_fitness < np.min([func(p) for p in best_swarm for best_swarm in self.swarms]):
                if self.swarm_size < self.max_swarm_size:
                    self.swarm_size += 1
            else:
                if self.swarm_size > self.min_swarm_size:
                    self.swarm_size -= 1

        final_best_swarm = min(self.swarms, key=lambda x: np.min([func(p) for p in x]))
        return min(final_best_swarm, key=lambda x: func(x))