import pickle
import networkx as nx

with open("cleaned_graph.pkl", "rb") as f:
    data = pickle.load(f)

G = nx.Graph()
G.add_nodes_from(data["nodes"])
G.add_edges_from(data["edges"])


for u, v in G.edges():
    if G.edges[u, v] == {}:  # empty attribute dict
        G.edges[u, v]['type'] = 'unknown'
        G.edges[u, v]['weight'] = 1.0

# Prepare and save as dict
data_cleaned = {
    "nodes": list(G.nodes(data=True)),
    "edges": list(G.edges(data=True))
}
with open("graph3.pkl", "wb") as f:
    pickle.dump(data_cleaned, f)

print("✅ Cleaned graph saved in expected format as 'cleaned_fixed_graph.pkl'")
