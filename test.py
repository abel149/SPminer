import pickle

with open("new_graph.pkl", "rb") as f:
    data = pickle.load(f)

# Save with protocol 4 and no NumPy objects
with open("g_s.pkl", "wb") as f:
    pickle.dump(data, f, protocol=4)
