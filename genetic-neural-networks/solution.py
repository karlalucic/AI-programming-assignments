import argparse, csv, random                     
import numpy as np                             


def load_data(path):
    """return X(NxD array) and y(N,) from csv -- (N rows, D features)"""
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)                    
        rows = [row for row in reader]

    X = []
    y = []
    for row in rows:                             
        X.append([float(v) for v in row[:-1]])
        y.append(float(row[-1]))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)   # convert str to float


def sigmoid(x):
    """hidden layers use it 
    the output layer stays linear (regression task)"""
    return 1.0 / (1.0 + np.exp(-x))

class NeuralNetwork:
    """simple fully connected network
    multiple hidden layers with sigmoid activation
    a linear output layer (no activation function)"""

    def __init__(self, sizes, weights=None, biases=None):
        #network architecture - list
        self.sizes = sizes
        
        # 2 init paths, first with provided weights and biases, and rand with small val from norm distrib

        if weights is not None and biases is not None:
            # clone path (elitism)
            self.weights = weights
            self.biases  = biases
            return

        # random init  N(0, 0.01^2)
        self.weights = []
        self.biases  = []
        for layer in range(len(sizes) - 1):
            in_dim  = sizes[layer]             # neurons feeding into this layer
            out_dim = sizes[layer + 1]         # neurons produced by this layer

            weight_matrix = np.random.normal(0.0, 0.01, (in_dim, out_dim))
            bias_row = np.random.normal(0.0, 0.01, (1, out_dim)) 

            self.weights.append(weight_matrix)
            self.biases.append(bias_row)

    def forward(self, x):
        """vector -> scalar prediction"""
        a = x.reshape(1, -1)                     # reshape input to row vector format
        for idx in range(len(self.weights) - 1):      # process through hidden layers with sigmoid act function
            z = np.dot(a, self.weights[idx]) + self.biases[idx]     # calculate the weighted sum of inputs + bias for each neuron in the curr layer
            a = sigmoid(z)
        # last layer (linear) = weighted sum + bias
        out = np.dot(a, self.weights[-1]) + self.biases[-1]
        return out.ravel()[0]                    # from matrix to scalar val for regression

    def mse(self, X, y):
        """mean squared error"""
        preds = np.array([self.forward(row) for row in X])  # Makes predictions for each input row
        diff  = y - preds                   # Computes the difference between true and predicted values
        return np.mean(diff ** 2)

    def clone(self):
        """deep copy of params, for elitsm"""
        new_w = [w.copy() for w in self.weights]
        new_b = [b.copy() for b in self.biases]
        return NeuralNetwork(self.sizes, new_w, new_b)  # passes the same arch, but diff weights and biases


class GeneticAlgo:
    """basic GA with roulette selection, arithmetic crossover, Gaussian mutate"""

    def __init__(self, sizes, pop_size, p_mut, sigma, elitism):
        self.sizes = sizes            # neural net arch
        self.pop_size = pop_size      # num of networks in pop
        self.p_mut = p_mut            # probability of mutation
        self.sigma = sigma            # mutation noise magnitude
        self.elitism = elitism        # num of top networks to preserve
        self.population = [NeuralNetwork(sizes) for n in range(pop_size)]   #creates an initial pop of rand initi neural networks

    def fitness(self, X, y):
        """return list of fitness"""
        fit = []
        for net in self.population:
            err = net.mse(X, y)
            fit.append(1.0 / err)
        return fit                              # Returns a list of fitness values corresponding to each network in the population

    def pick_parent(self, fitness):                     # networks with higher fitness have higher prob of being selected as parents
        """fitness proportionate selection -> index      
        (selecting individuals in ga)"""
        total = sum(fitness)
        r = random.random() * total     # picks a random point between 0 and the total fitness
        s = 0.0
        for i, f in enumerate(fitness):
            s += f
            if r <= s:                      # when the accumulated value exceeds the random point r -> that individual is selected
                return i
        return len(fitness) - 1                 # numerical fallback

    def crossover(self, pa, pb):
        """child = average of parents"""
        new_w = [(wa + wb) * 0.5 for wa, wb in zip(pa.weights, pb.weights)]
        new_b = [(ba + bb) * 0.5 for ba, bb in zip(pa.biases,  pb.biases)]
        return NeuralNetwork(self.sizes, new_w, new_b)      # same arh, new avg weights and biases


    def mutate(self, net):
        """add Gaussian (N(0,o^2)) noise with prob p_mut to every param"""
        for layer_idx in range(len(net.weights)):                 # each layer
            layer = net.weights[layer_idx]
            rows, cols = layer.shape
            for i in range(rows):
                for j in range(cols):
                    if random.random() < self.p_mut:              # coin-flip
                        noise = random.gauss(0.0, self.sigma)     # only applies mutation if the random check passes
                        layer[i, j] += noise

        for layer_idx in range(len(net.biases)):
            layer = net.biases[layer_idx]                        
            cols = layer.shape[1]
            for j in range(cols):
                if random.random() < self.p_mut:
                    noise = random.gauss(0.0, self.sigma)
                    layer[0, j] += noise

    def step(self, X, y):
        """one generation"""
        fit = self.fitness(X, y)

        # keep elites
        elite_idx = np.argsort(fit)[-self.elitism:]             # Takes the last elitism indices (highest values)
        new_pop = [self.population[i].clone() for i in elite_idx]

        # fill the rest
        while len(new_pop) < self.pop_size:
            pa = self.population[self.pick_parent(fit)]
            pb = self.population[self.pick_parent(fit)]
            child = self.crossover(pa, pb)          # Create children through crossover
            self.mutate(child)
            new_pop.append(child)

        self.population = new_pop

    def best_net(self, X, y):
        """return best net and its mse"""
        fit = self.fitness(X, y)
        idx = int(np.argmax(fit))
        net = self.population[idx]
        return net, net.mse(X, y)


def parse_architecture(key, d_in):
    if key == '5s':
        return [d_in, 5, 1]
    if key == '20s':
        return [d_in, 20, 1]
    if key == '5s5s':
        return [d_in, 5, 5, 1]


def main():
    parser = argparse.ArgumentParser(description='simple GA + numpy NN')
    parser.add_argument("--train", type=str)
    parser.add_argument("--test", type=str)
    parser.add_argument("--nn", type=str)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--elitism", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--K", type=float)
    parser.add_argument("--iter", type=int)
    args = parser.parse_args()

    X_train, y_train = load_data(args.train)
    X_test, y_test = load_data(args.test)

    # GA setup
    arch = parse_architecture(args.nn, X_train.shape[1])       #Converts a string code into a nn structure
    ga = GeneticAlgo(arch, args.popsize, args.p, args.K, args.elitism)


    for it in range(1, args.iter + 1):
        ga.step(X_train, y_train)
        if it % 2000 == 0:                  #Every 2k iterations, prints the curr best network train error
            net, err = ga.best_net(X_train, y_train)
            print(f"[Train error @{it}]: {err:.6f}")

    # evaluate the best network on the test data
    net, test_err = ga.best_net(X_test, y_test)
    # report final mse on test data
    print(f"[Test error]: {test_err:.6f}")


if __name__ == "__main__":
    main()
