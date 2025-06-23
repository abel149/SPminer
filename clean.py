import pickle
import networkx as nx

# Load your cleaned .pkl file
with open("graph3.pkl", "rb") as f:
    data = pickle.load(f)

# Convert if needed
if isinstance(data, dict) and "nodes" in data and "edges" in data:
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
else:
    G = data if isinstance(data, nx.Graph) else None

if not isinstance(G, nx.Graph):
    print("❌ Not a valid NetworkX graph.")
    exit()

print("✅ Loaded NetworkX graph.")
print("Checking edge attributes...\n")

empty_count = 0
for u, v, attrs in G.edges(data=True):
    if not attrs or not isinstance(attrs, dict) or attrs == {}:
        print(f"⚠️ Edge ({u}, {v}) has EMPTY attributes: {attrs}")
        empty_count += 1
    else:
        missing_keys = []
        if 'type' not in attrs:
            missing_keys.append('type')
        if 'weight' not in attrs:
            missing_keys.append('weight')
        if missing_keys:
            print(f"⚠️ Edge ({u}, {v}) missing keys: {missing_keys}")

print(f"\n✅ Done. Total edges with empty or missing attributes: {empty_count}")
