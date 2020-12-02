import numpy as np
from Tree import Tree

#Storing computed values in a dictionary
dict = {}


def calculate_likelihood(tree_topology, theta, beta):
    #Initialize tree, likelihood, K
    t = Tree()
    t.load_tree_from_direct_arrays(tree_topology, theta)
    likelihood = 0
    K = t.k

    #Start iterative calculations for p(beta|tree,theta) from root, clear dict for each sample
    for i in range(K):
        likelihood += subProb(t.root, i, theta, tree_topology, beta, K)*theta[0][i]
        dict.clear()
    return likelihood


def subProb(Node, i, theta, tree_topology, beta, K):
    #Initalize index, sums
    probInd = str(str(i) + str(Node.name))
    lSum = 0
    rSum = 0

    #If value calculated, access from dict
    if probInd in dict:
        return dict[probInd]
    #When leaf-node is reached, set probability to 1 if value aligns, 0 otherwise
    if not Node.descendants:
        if beta[int(Node.name)] == i:
            return 1
        else:
            return 0

    #Initalize children
    lChild = Node.descendants[0]
    rChild = Node.descendants[1]
    #Iterative calculations for p below Node
    for j in range(K):
        lSum += theta[int(lChild.name)][i][j]*subProb(lChild, j, theta, tree_topology, beta, K)
        rSum += theta[int(rChild.name)][i][j]*subProb(rChild, j, theta, tree_topology, beta, K)
    subVal = lSum*rSum
    dict[probInd] = subVal
    return subVal


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")
    print("\n1. Load tree data from file and print it\n")

    filename = "q2_2_small_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()

    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))
    print("\n2. Calculate likelihood of each FILTERED sample\n")
    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)

if __name__ == "__main__":
    main()