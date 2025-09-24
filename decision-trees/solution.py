import math
import argparse, csv
from collections import Counter     # quick freq counting


def load_data(path):
    """return header and list of row dicts"""
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)                               #first row is the header (feature names + class label)
        rows = [dict(zip(header, row)) for row in reader]     # each row is a dict mapping column name ->value {feature:value}
    return header, rows   

def mcv(rows, label):
    """most common value in class label in rows"""
    """leaf label when D = empty | fallback during prediction on an unseen attr val"""
    counts = Counter(row[label] for row in rows)    # label -> frequency 
    max_freq = max(counts.values())                 #highest frequency
    best = [lab for lab, n in counts.items() if n == max_freq]      # to find labels with maximum frequency
    return sorted(best)[0]                          # alphabetical

def entropy(rows, label):
    total = len(rows)
    if total == 0:
        return 0.0 
    counts = Counter(row[label] for row in rows)        # counts the frequency of each unique class label
    # sum -p * log2(p) for every class
    return sum(-(c/total) * math.log2(c/total) for c in counts.values())

def information_gain(rows, attribute, label):
    """reduction in entropy when partitioning D by a given feature"""
    base_E = entropy(rows, label)
    
    # partition rows by each possible value "v" of the attribute
    partitions = {}
    for row in rows:
        attr_val = row[attribute]
        
        if attr_val not in partitions:
            partitions[attr_val] = []
            
        partitions[attr_val].append(row)

    expected_E = 0.0
    
    for partition in partitions.values():
        expected_E += (len(partition) / len(rows)) * entropy(partition, label)
        
    return base_E - expected_E

class Node:
    def __init__(self, attribute=None, label=None, mcv=None):
        self.attribute = attribute      # the splitting feature(None for leaves)
        self.label = label              # class label stored in leaves
        self.mcv = mcv                  # most common label at this node (for prediction fallback)
        self.children = {}              # dict feature value-> child node
        
def id3(D, D_parent, features, label, depth, depth_limit):
    # D - current dataset (list of dicts, each row is a dict)
    # D_parent - parent dataset (we use it when D is empty)
    # features - list of attributes still available for partitioning
    # label - name of class column (in last csv column)
    # depth - current depth
    #depth_limit - optional (for final task)
    
    # if D=empty -> return Leaf with parent's mcv)
    if len(D) == 0:
        return Node(label=mcv(D_parent, label))
    
    # used both for potential leaf nodes and as a fallback
    curr_mcv = mcv(D, label)    # v <- argmax_v (D_y=y)
    
    # check if all examples have same class
    if all(row[label] == curr_mcv for row in D):
        return Node(label=curr_mcv, mcv=curr_mcv)
    
    # stop conditions (stop and create a leaf node if)
    no_features = len(features) == 0
    depth_limit_reached = depth_limit is not None and depth >= depth_limit
    if no_features or depth_limit_reached:
        return Node(label=curr_mcv, mcv=curr_mcv)
    
    #choosing an attribute with highest information gain
    gains = [(information_gain(D, atr, label), atr) for atr in features]    # calculate ig for each feature
    top_g = max(gain for (gain, feature) in gains)      # feature with highest gain
    candidates = [a for g, a in gains if g == top_g]    # if more attrs have the same ig, then alphabetical 
    top_attr  = sorted(candidates)[0]
    
    # create internal node using most discriminative attr, thern recurse on each value of top_attr
    node = Node(attribute=top_attr, mcv=curr_mcv)
    values = sorted({row[top_attr] for row in D})   # V(x)
    other = [f for f in features if f != top_attr]  # filtering out the top_attr so it won't be considered again in further ops
    
    for v in values:
        subtree = [r for r in D if r[top_attr] == v]        # a subset of data where the attr = value
        t = id3(subtree, D, other, label, depth + 1, depth_limit)
        node.children[v] = t                                # add the subtree as a child of the curr node
        
    return node
    
def walk(node, path, level):
        if node.attribute is None:
            # leaf - print the path so far than the label
            print(" ".join(path), node.label)
            return
        #for each child add the step description and the label
        for value, child in sorted(node.children.items()):
            step = f"{level}:{node.attribute}={value}"
            walk(child, path + [step], level+1)

def print_tree(root):
    print("[BRANCHES]:")
    walk(root, [], 1)
    
def predict(root, row):
    """make a prediction for a single row"""
    node = root
    # at each node look for the attribute in the curr row
    while node.attribute is not None:
        val = row.get(node.attribute)
        if val in node.children:    # if the value is known -> follow that branch
            node = node.children[val]
        else:
            return node.mcv     # unseen attr val, return most common class in curr node as fallback
    return node.label           # reached leaf -> return class label in the leaf

def accuracy(preds, truths):
    correct = sum(p == t for p, t in zip(preds, truths))    # count how many predictions match the true labels
    accurate = correct / len(truths) if truths else 0.0
    print(f"[ACCURACY]: {accurate:.5f}")
    

def confusion_matrix(preds, truths):
    """matrix where rows = true labels and columns = predicted labels"""
    labels = sorted(set(preds) | set(truths))
    index = {}
    for i in range(len(labels)):
        index[labels[i]] = i
    # build an nxn zero matrix
    matrix = []
    for _ in range(len(labels)):
        row = []
        for _ in range(len(labels)):
            row.append(0)
        matrix.append(row)
    
    for p, t in zip(preds, truths):
        matrix[index[t]][index[p]] += 1
        
    print("[CONFUSION_MATRIX]:")
    for row in matrix:
        print(" ".join(map(str, row)))
    

def main():
    parser = argparse.ArgumentParser(description='id3')
    parser.add_argument("train_ds", type=str)
    parser.add_argument("test_ds", type=str)
    parser.add_argument("depth_limit", type=int, nargs='?')
    args = parser.parse_args()
    
    train_header, train_rows = load_data(args.train_ds)
    test_header, test_rows = load_data(args.test_ds)

    label = train_header[-1]        # last column -> class
    features = train_header[:-1]    # other -> attributes
    
    
    decision_tree = id3(train_rows, train_rows, features, label, depth=0, depth_limit=args.depth_limit)
    
    print_tree(decision_tree)
    
    preds = [predict(decision_tree, row) for row in test_rows]
    print("[PREDICTIONS]:", " ".join(preds)) 
    truths = [row[label] for row in test_rows]
    
    accuracy(preds, truths)
    confusion_matrix(preds, truths)    


if __name__ == "__main__":
    main()