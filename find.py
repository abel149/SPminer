import pickle
import networkx as nx

with open("cleaned_graph.pkl", "rb") as f:
    data = pickle.load(f)

G = nx.Graph()
G.add_nodes_from(data["nodes"])
G.add_edges_from(data["edges"])

# Fix empty edge attributes
for u, v in G.edges():
    if not G.edges[u, v]:  # Empty dict
        G.edges[u, v]['type'] = 'unknown'
        G.edges[u, v]['weight'] = 1.0

# Save the fixed graph
with open("cleaned_fixed_graph.pkl", "wb") as f:
    pickle.dump(G, f)

print("✅ Fixed and saved as 'cleaned_fixed_graph.pkl'")
