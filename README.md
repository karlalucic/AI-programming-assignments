# AI Programming Assignments

## Repository Structure

```
.
├── state-space-search/
│   └── solution.py          # Search algorithms implementation
├── refutation-resolution/
│   └── solution.py          # Logic reasoning system
├── decision-trees/
│   └── solution.py          # ID3 algorithm implementation
├── genetic-neural-networks/
│   └── solution.py          # Evolutionary neural network training
└── README.md
```

## Assignment Overview

### State Space Search
Implementation of search algorithms for pathfinding and puzzle-solving problems.

**Algorithms**: Breadth-First Search (BFS), Uniform Cost Search (UCS), A* Algorithm
**Features**: Heuristic validation, support for 8-puzzle and map navigation

```bash
python solution.py --alg [bfs|ucs|astar] --ss <state_file> --h <heuristic_file>
python solution.py --ss <state_file> --h <heuristic_file> --check-optimistic
```

### Refutation Resolution
Automated reasoning system using propositional logic and resolution.

**Features**: Resolution algorithm, interactive cooking assistant with knowledge base operations

```bash
python solution.py resolution <clauses_file>
python solution.py cooking <clauses_file> <commands_file>
```

### Decision Trees
ID3 algorithm implementation for classification tasks.

**Features**: Information gain calculation, tree depth limiting, accuracy evaluation

```bash
python solution.py <train_file> <test_file> [depth_limit]
```

### Genetic Neural Networks
Neural network training using evolutionary algorithms instead of backpropagation.

**Features**: Feedforward networks, genetic optimization, function approximation

```bash
python solution.py --train <train_file> --test <test_file> --nn [5s|20s|5s5s] \
                   --popsize 10 --elitism 1 --p 0.1 --K 0.1 --iter 10000
```

## Requirements

- Python 3.7.4+
- Standard library only (NumPy allowed for genetic neural networks)
- UTF-8 encoding

## Input Formats

- **State Space Search**: Custom text format for states and transitions
- **Refutation Resolution**: CNF clauses with disjunction 'v' and negation '¬'  
- **Decision Trees & Genetic Neural Networks**: CSV format with headers

Each solution handles the specific file formats and command-line arguments as required by the course autograder system.
