import pickle

# Path to your .pkl file
pkl_file = "graph.pkl"

# Load the .pkl file
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

# Check the type
print("Type of loaded object:", type(data))

# If it's a NetworkX graph
if isinstance(data, dict):
    # Sometimes pickled graphs are stored as dict with 'nodes' and 'edges'
    print("Keys in the dictionary:", data.keys())
    if 'nodes' in data:
        print("Sample nodes:", list(data['nodes'])[:5])
    if 'edges' in data:
        print("Sample edges:", list(data['edges'])[:5])
elif 'networkx' in str(type(data)).lower():
    # Direct NetworkX graph
    print("Nodes:", list(data.nodes())[:5])
    print("Edges:", list(data.edges(data=True))[:5])
else:
    # Unknown format
    print(data)
