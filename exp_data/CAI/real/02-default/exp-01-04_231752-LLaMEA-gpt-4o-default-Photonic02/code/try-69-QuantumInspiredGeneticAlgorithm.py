import numpy as np

class QuantumInspiredGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.population = None
        self.best_solution = None
        self.best_score = np.inf
        self.quantum_chromosomes = None
        self.rotation_angle = np.pi / 4

    def _initialize_population(self, lb, ub):
        self.quantum_chromosomes = np.random.rand(self.population_size, self.dim, 2)  # Quantum representation
        self.population = self._quantum_to_real(self.quantum_chromosomes, lb, ub)

    def _quantum_to_real(self, quantum_chromosomes, lb, ub):
        # Convert quantum representation to real values using probabilistic interpretation
        angles = np.arctan2(quantum_chromosomes[:,:,1], quantum_chromosomes[:,:,0])
        probabilities = (np.sin(angles)**2)
        return lb + probabilities * (ub - lb)

    def _quantum_rotation(self, idx, best_idx):
        # Apply quantum rotation gate based on the best solution found
        angle_diff = np.arctan2(self.quantum_chromosomes[best_idx,:,1], self.quantum_chromosomes[best_idx,:,0]) - \
                     np.arctan2(self.quantum_chromosomes[idx,:,1], self.quantum_chromosomes[idx,:,0])
        rotation = np.clip(angle_diff, -self.rotation_angle, self.rotation_angle)
        new_angles = np.arctan2(self.quantum_chromosomes[idx,:,1], self.quantum_chromosomes[idx,:,0]) + rotation
        self.quantum_chromosomes[idx,:] = np.array([np.cos(new_angles), np.sin(new_angles)]).T

    def _quantum_crossover(self, parent1, parent2):
        # Perform quantum crossover by averaging the angles
        angles1 = np.arctan2(parent1[:,:,1], parent1[:,:,0])
        angles2 = np.arctan2(parent2[:,:,1], parent2[:,:,0])
        new_angles = (angles1 + angles2) / 2
        return np.array([np.cos(new_angles), np.sin(new_angles)]).T

    def _quantum_mutation(self, chromosome):
        # Perform quantum mutation by adding a small random angle
        mutation_angle = np.random.uniform(-self.rotation_angle/10, self.rotation_angle/10, self.dim)
        angles = np.arctan2(chromosome[:,1], chromosome[:,0]) + mutation_angle
        chromosome[:] = np.array([np.cos(angles), np.sin(angles)]).T

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            scores = np.array([func(ind) for ind in self.population])
            eval_count += self.population_size

            best_idx = np.argmin(scores)
            if scores[best_idx] < self.best_score:
                self.best_score = scores[best_idx]
                self.best_solution = self.population[best_idx]

            # Evolutionary process
            new_quantum_chromosomes = np.zeros_like(self.quantum_chromosomes)
            for i in range(self.population_size):
                self._quantum_rotation(i, best_idx)
                partner_idx = np.random.randint(self.population_size)
                new_quantum_chromosomes[i] = self._quantum_crossover(self.quantum_chromosomes[i:i+1], 
                                                                     self.quantum_chromosomes[partner_idx:partner_idx+1])
                self._quantum_mutation(new_quantum_chromosomes[i])

            self.quantum_chromosomes = new_quantum_chromosomes
            self.population = self._quantum_to_real(self.quantum_chromosomes, self.lb, self.ub)

        return self.best_solution, self.best_score