import pickle
import networkx as nx

def clean_edge_attrs(attrs):
    return {str(k): v for k, v in attrs.items() if isinstance(k, str) and isinstance(v, (int, float, str))}

def clean_node_attrs(attrs):
    return {str(k): v for k, v in attrs.items() if isinstance(k, str) and isinstance(v, (int, float, str))}

with open("graph.pkl", "rb") as f:
    data = pickle.load(f)

graph = nx.Graph()
graph.add_nodes_from(data['nodes'])
graph.add_edges_from(data['edges'])

# Clean node attributes
for node in graph.nodes():
    attrs = graph.nodes[node]
    graph.nodes[node].clear()
    graph.nodes[node].update(clean_node_attrs(attrs))

# Clean edge attributes
for u, v in graph.edges():
    attrs = graph.edges[u, v]
    graph.edges[u, v].clear()
    graph.edges[u, v].update(clean_edge_attrs(attrs))

# Save the cleaned graph to a new pkl
with open("cleaned_graph.pkl", "wb") as f:
    cleaned_data = {
        "nodes": list(graph.nodes(data=True)),
        "edges": list(graph.edges(data=True))
    }
    pickle.dump(cleaned_data, f)

print("✅ Cleaned graph saved to cleaned_graph.pkl")
