import pandas as pd
import numpy as np
import networkx as nx

df = pd.read_csv("/content/SP500_Closing_Prices-NEW - Copy.csv", header=None)
print("Original Shape:", df.shape)

df = df.fillna(method='ffill')
df = df.fillna(method='bfill')
df = df.fillna(df.mean())
print("Missing values after cleaning:", df.isnull().sum().sum())

returns_full = df.pct_change().dropna()
market_return_full = returns_full.mean(axis=1)
csad_full = (returns_full.sub(market_return_full, axis=0).abs()).mean(axis=1)

csad_df = pd.DataFrame({'CSAD': csad_full})
csad_df.to_csv("herding_csad_full.csv", index=False)
print("Full-data CSAD saved to herding_csad_full.csv")

df_70 = df.iloc[:int(0.70 * len(df))]
returns = df_70.pct_change().dropna()
T, N = returns.shape
print("70% Data Shape:", returns.shape)

market_return = returns.mean(axis=1)
csad = (returns.sub(market_return, axis=0).abs()).mean(axis=1)

rho = []
CSAD_centered = csad - csad.mean()

for col in returns.columns:
    Ri = returns[col]
    Ri_centered = Ri - Ri.mean()
    numerator = np.sum(Ri_centered * CSAD_centered)
    denominator = np.sqrt(np.sum(Ri_centered**2)) * np.sqrt(np.sum(CSAD_centered**2))
    rho_i = numerator / denominator if denominator != 0 else 0
    rho.append(rho_i)

rho = np.array(rho)
print("Rho min:", np.min(rho), "Rho max:", np.max(rho))

rho_norm = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))

rho_diff = np.abs(rho_norm[:, None] - rho_norm[None, :])
S = 1 - rho_diff

tau = np.percentile(S, 25)
print("Adaptive Threshold (tau):", round(tau, 4))

A = (S > tau).astype(int)
np.fill_diagonal(A, 0)

adj_matrix = pd.DataFrame(A)
adj_matrix.to_csv("herding_adjacency_matrix_25%.csv", index=False)
print("Herding adjacency matrix saved.")
print("Adjacency Matrix Shape:", adj_matrix.shape)

num_nodes = A.shape[0]
ones_count = np.sum(A == 1)
zeros_count = np.sum(A == 0)
unique_edges = ones_count // 2
max_possible_edges = num_nodes * (num_nodes - 1) / 2
density = unique_edges / max_possible_edges
sparsity = 1 - density
degrees = np.sum(A, axis=1)
avg_degree = np.mean(degrees)
max_degree = np.max(degrees)
min_degree = np.min(degrees)

G = nx.from_numpy_array(A)
num_components = nx.number_connected_components(G)
avg_clustering = nx.average_clustering(G)

print("\n========== Herding Graph Statistics ==========")
print("Number of Nodes:", num_nodes)
print("Total 1s (Edges counted twice):", ones_count)
print("Total 0s:", zeros_count)
print("Unique Undirected Edges:", unique_edges)
print("Graph Density:", round(density, 4))
print("Graph Sparsity:", round(sparsity, 4))
print("Average Degree:", round(avg_degree, 2))
print("Maximum Degree:", max_degree)
print("Minimum Degree:", min_degree)
print("Connected Components:", num_components)
print("Average Clustering Coefficient:", round(avg_clustering, 4))
print("=============================================")
