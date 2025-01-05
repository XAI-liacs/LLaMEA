import numpy as np

class EnhancedCooperativeQuantumSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.num_particles = max(10, min(30, budget // 10))
        self.swarms = [self.initialize_swarm() for _ in range(self.num_swarms)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
    
    def initialize_swarm(self):
        swarm = {
            'particles': None,
            'velocities': None,
            'personal_best_positions': None,
            'personal_best_fitness': np.full(self.num_particles, float('inf')),
            'local_best_position': None,
            'local_best_fitness': float('inf')
        }
        return swarm
    
    def initialize_particles(self, swarm, lb, ub):
        swarm['particles'] = lb + (ub - lb) * np.random.rand(self.num_particles, self.dim)
        swarm['velocities'] = np.random.randn(self.num_particles, self.dim) * 0.1
        swarm['personal_best_positions'] = np.copy(swarm['particles'])
    
    def evaluate_particles(self, swarm, func):
        fitness = np.array([func(p) for p in swarm['particles']])
        for i, f in enumerate(fitness):
            if f < swarm['personal_best_fitness'][i]:
                swarm['personal_best_fitness'][i] = f
                swarm['personal_best_positions'][i] = swarm['particles'][i]
            if f < swarm['local_best_fitness']:
                swarm['local_best_fitness'] = f
                swarm['local_best_position'] = swarm['particles'][i]
            if f < self.global_best_fitness:
                self.global_best_fitness = f
                self.global_best_position = swarm['particles'][i]
        return fitness
    
    def update_velocities_and_positions(self, swarm, lb, ub):
        inertia_weight = 0.7
        cognitive_const = 2.0
        social_const = 2.0
        constriction_factor = 0.5
        
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_term = cognitive_const * r1 * (swarm['personal_best_positions'][i] - swarm['particles'][i])
            social_term = social_const * r2 * (swarm['local_best_position'] - swarm['particles'][i])
            swarm['velocities'][i] = (inertia_weight * swarm['velocities'][i] +
                                      cognitive_term + social_term)
            swarm['particles'][i] += constriction_factor * swarm['velocities'][i]
        
        swarm['particles'] = np.clip(swarm['particles'], lb, ub)
    
    def adaptive_quantum_behaviors(self, swarm, lb, ub):
        for i in range(self.num_particles):
            if np.random.rand() < 0.1:  # Quantum teleportation probability
                swarm['particles'][i] = lb + (ub - lb) * np.random.rand(self.dim)
            elif np.random.rand() < 0.1:  # Quantum coherence
                coherence_vector = lb + (ub - lb) * np.random.rand(self.dim)
                swarm['particles'][i] = 0.5 * (swarm['particles'][i] + coherence_vector)
            swarm['particles'][i] = np.clip(swarm['particles'][i], lb, ub)
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        for swarm in self.swarms:
            self.initialize_particles(swarm, lb, ub)
        
        evaluations = 0
        while evaluations < self.budget:
            for swarm in self.swarms:
                self.evaluate_particles(swarm, func)
                evaluations += self.num_particles
                if evaluations >= self.budget:
                    break
                self.update_velocities_and_positions(swarm, lb, ub)
                self.adaptive_quantum_behaviors(swarm, lb, ub)
        
        return self.global_best_position, self.global_best_fitness