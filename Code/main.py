from EA import EA

if __name__ == '__main__':

    ea = EA(generations=100, p_size=128, num_offspring=256, mutation_rate=0.01)
    ea.run(1)
    ea.plot()
    ea.gene_to_midi('demo.mid')
