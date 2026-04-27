import numpy as np

class EvolutionSystem:
    def __init__(self, population_size=50, elite_count=5, mutation_rate=0.05, 
                 mutation_strength=0.1, crossover_points=2):
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = max(5, int(population_size * 0.07))
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_points = crossover_points

    def tournament_select(self, agents, fitnesses):
        """Pick a random subset of agents and return the best one."""
        idx = np.random.choice(len(agents), self.tournament_size, replace=False)
        best_idx = idx[np.argmax(fitnesses[idx])]
        return agents[best_idx]

    def n_point_crossover(self, w1, w2):
        """Splice two parent genomes together."""
        pts = sorted(np.random.choice(len(w1) - 1, self.crossover_points, replace=False) + 1)
        child = w1.copy()
        use_w2 = False
        prev = 0
        for pt in pts + [len(w1)]:
            if use_w2:
                child[prev:pt] = w2[prev:pt]
            use_w2 = not use_w2
            prev = pt
        return child

    def mutate(self, weights):
        """Randomly alter a small percentage of the weights."""
        mask = np.random.random(len(weights)) < self.mutation_rate
        weights[mask] += np.random.randn(mask.sum()) * self.mutation_strength
        return weights

    def evolve(self, agents, fitnesses):
        """Rank the population, breed children, and overwrite the agents' brains."""
        fitnesses = np.array(fitnesses)
        
        # Sort indices by fitness (lowest to highest), then take the top 'elite_count'
        elite_idx = np.argsort(fitnesses)[-self.elite_count:]
        
        # 1. Elites survive exactly as they are (Elitism)
        new_weights = [agents[i].get_weights_flat() for i in elite_idx]

        # 2. Breed the rest of the population
        while len(new_weights) < self.population_size:
            p1 = self.tournament_select(agents, fitnesses)
            p2 = self.tournament_select(agents, fitnesses)
            
            # Extract 1D numpy arrays from PyTorch models
            w1 = p1.get_weights_flat()
            w2 = p2.get_weights_flat()
            
            # Crossover and Mutate
            child_weights = self.n_point_crossover(w1, w2)
            child_weights = self.mutate(child_weights)
            
            new_weights.append(child_weights)

        # 3. Inject new weights back into the PyTorch agents
        for agent, w in zip(agents, new_weights):
            agent.set_weights_flat(w)
            
        print(f"Evolution complete. Top fitness score: {fitnesses[elite_idx[-1]]:.1f}")