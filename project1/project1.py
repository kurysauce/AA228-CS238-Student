import sys
import networkx as nx
import pandas as pd
import numpy as np

# Helper function to write the graph structure in the desired format
def write_gph(dag, idx2names, filename):
    print("Writing graph to file...")
    print(f"Edges in graph: {list(dag.edges())}")
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

# Helper to compute conditional probability of a node given its parents
def compute_conditional_probability(node, parents, data, alpha=1):
    """
    Compute the conditional probability of a node given its parents.
    Returns a conditional probability table (CPT) as a pandas DataFrame.
    """
    if parents:
        counts = data.groupby(parents + [node]).size().unstack(fill_value=0)
        counts += alpha
        cpt = counts.div(counts.sum(axis=1), axis=1)
        cpt = cpt.fillna(1e-10)
    else:
        cpt = data[node].value_counts(normalize=True)
    
    return cpt

# Function to calculate the Bayesian score
def compute_bayesian_score(dag, data, alpha=1):
    score = 0
    nodes = data.columns.tolist()

    for node in nodes:
        parents = list(dag.predecessors(node))
        cpt = compute_conditional_probability(node, parents, data, alpha=alpha)
        node_data = data[node]

        print(f"Calculating score for node '{node}' with parents {parents}")
        print(f"Conditional Probability Table (CPT) for {node}:")
        print(cpt.head())

        if parents:
            parent_data = data[parents]
            for i, row in parent_data.iterrows():
                parent_tuple = tuple(row)
                if parent_tuple in cpt.index:
                    prob = cpt.loc[parent_tuple][node_data[i]]
                else:
                    prob = 1e-10
                score += np.log(prob)
        else:
            for value in node_data:
                prob = cpt.get(value, 1e-10)
                score += np.log(prob)
        
        print(f"Score after node '{node}': {score}\n")

    return score

# Function to implement the search for the best graph structure (with relaxed constraints)
def k2_algorithm(data, node_order, max_parents, score_improvement_threshold=1e-200):
    """K2 algorithm for learning Bayesian network structure with relaxed constraints."""
    nodes = data.columns.tolist()
    
    # Initialize the graph as an empty directed graph
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    
    for i, node in enumerate(node_order):
        current_parents = []
        best_score = compute_bayesian_score(dag, data)
        
        print(f"\nProcessing node '{node}' with current score: {best_score}")
        
        for potential_parent in node_order[:i]:
            if len(current_parents) < max_parents:
                dag.add_edge(potential_parent, node)
                new_score = compute_bayesian_score(dag, data)
                
                print(f"Trying to add edge {potential_parent} -> {node}, new score: {new_score}")
                
                if new_score > best_score + score_improvement_threshold:  # Relaxed threshold
                    best_score = new_score
                    current_parents.append(potential_parent)
                    print(f"Edge {potential_parent} -> {node} added")
                else:
                    dag.remove_edge(potential_parent, node)
                    print(f"Edge {potential_parent} -> {node} removed")
    
    return dag

# Main function to handle input/output
def compute(infile, outfile):
    data = pd.read_csv(infile)
    idx2names = {i: name for i, name in enumerate(data.columns)}

    # Define node order and allow more parents (relaxed constraint)
    node_order = ['age', 'sex', 'fare', 'passengerclass', 'portembarked', 'numparentschildren', 'numsiblings', 'survived']
    max_parents = 2 # Allowing up to 4 parents per node
    score_improvement_threshold = 1e-4  # Relax the score improvement threshold

    best_dag = k2_algorithm(data, node_order, max_parents, score_improvement_threshold)

    # Output the best graph structure to a file
    write_gph(best_dag, idx2names, outfile)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == '__main__':
    main()
