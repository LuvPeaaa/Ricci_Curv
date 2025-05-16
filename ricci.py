import numpy as np
import pandas as pd
import os
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
!pip install dtaidistance
!pip install --upgrade dtaidistance
from dtaidistance import dtw
import matplotlib.pyplot as plt
!pip install GraphRicciCurvature
from GraphRicciCurvature.OllivierRicci import OllivierRicci
!pip install faiss-cpu
import faiss


def load_data(file_path):
    """
    Load time series data from the given file path.
    Assumes CSV or similar tabular format.
    """
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        else:
            try:
                data = pd.read_csv(file_path)
            except Exception:
                data = pd.read_excel(file_path)

        print(f"Successfully loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_features_sliding_window(data, window_size, step_size=4):
    print("Starting feature extraction using sliding window...")
    features = []
    timestamps = []

    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        values = data[numeric_cols].values
    else:
        values = data

    total_windows = (len(values) - window_size) // step_size + 1
    for idx, i in enumerate(range(0, len(values) - window_size + 1, step_size)):
        window = values[i:i+window_size]
        features.append(window.flatten())
        timestamps.append(i)

        # Show progress every 10%
        if idx % max(1, total_windows // 10) == 0:
            print(f"  Extracted window {idx+1}/{total_windows}")

    print("Feature extraction complete.")
    return features, timestamps


def calculate_distance(feature_i, feature_j, method='euclidean'):
    """
    Calculate distance between two feature vectors using specified method.
    """
    if method == 'euclidean':
        return euclidean(feature_i, feature_j)
    elif method == 'cosine':
        # Convert cosine similarity to distance
        similarity = cosine_similarity([feature_i], [feature_j])[0][0]
        return 1 - similarity
    elif method == 'dtw':
        # DTW expects 1D sequences, so reshape if needed
        f_i = feature_i.reshape(-1, 1) if feature_i.ndim == 1 else feature_i
        f_j = feature_j.reshape(-1, 1) if feature_j.ndim == 1 else feature_j
        return dtw.distance(f_i, f_j)
    else:
        raise ValueError(f"Unsupported distance method: {method}")

def build_graph(features, timestamps, distance_method='euclidean', epsilon=0.05):
    print("Starting graph construction using pairwise distance...")
    G = nx.Graph()

    for i, ts in enumerate(timestamps):
        G.add_node(i, timestamp=ts)

    total_pairs = len(features) * (len(features) - 1) // 2
    pair_count = 0
    reported = 0

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            distance = calculate_distance(features[i], features[j], method=distance_method)
            pair_count += 1
            if distance < epsilon:
                G.add_edge(i, j, weight=distance)

            # Show progress every 10%
            if pair_count >= (total_pairs * (reported + 1)) // 10:
                print(f"  Processed {pair_count}/{total_pairs} pairs")
                reported += 1

    print(f"Graph construction complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def build_graph_approximate_knn(features, timestamps, distance_method='cosine', k_neighbors=5):
    print("Starting approximate kNN graph construction using FAISS...")
    if distance_method != 'cosine':
        raise ValueError("Approximate kNN is currently only supported for cosine distance")

    features = np.array(features).astype('float32')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_normalized = features / (norms + 1e-10)

    d = features_normalized.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(features_normalized)

    print("  Running kNN search...")
    distances, indices = index.search(features_normalized, k_neighbors + 1)
    G = nx.Graph()

    for i, ts in enumerate(timestamps):
        G.add_node(i, timestamp=ts)

    print("  Building edges from kNN results...")
    for i in range(len(features)):
        for j_idx in range(1, k_neighbors + 1):
            j = indices[i, j_idx]
            if i != j:
                sim = distances[i, j_idx]
                cosine_dist = 1 - sim
                G.add_edge(i, j, weight=cosine_dist)

        if i % max(1, len(features) // 10) == 0:
            print(f"    Processed node {i+1}/{len(features)}")

    print(f"Graph construction complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (Approximate kNN).")
    return G

def calculate_ricci_curvature(G):
    if G.number_of_edges() == 0:
        print("Graph has no edges. Cannot calculate Ricci curvature.")
        return None

    if not nx.is_connected(G):
        print("Warning: Graph is not connected. Working with the largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    print("Starting Ricci curvature computation...")
    orc = OllivierRicci(G, alpha=0.5, verbose="TRACE")
    orc.compute_ricci_curvature()
    print("Ricci curvature computation complete.")
    return orc.G


def visualize_graph(G, title="Time Series Graph"):
    """
    Visualize the graph with edges colored by Ricci curvature.
    """
    plt.figure(figsize=(12, 8))

    # Use spring layout for better visualization
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue')

    edges = G.edges()
    if 'ricciCurvature' in G[list(edges)[0][0]][list(edges)[0][1]]:
        edge_colors = [G[u][v]['ricciCurvature'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5, edge_color=edge_colors, edge_cmap=plt.cm.jet)

        # Add a colorbar
        cax = plt.axes([0.9, 0.1, 0.03, 0.8])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        sm.set_array(edge_colors)
        plt.colorbar(sm, cax=cax, label='Ricci Curvature')
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_time_series(file_path, window_size=20, step_size=4,
                        distance_method='euclidean', epsilon=0.05, visualize=True):
    """
    Complete analysis pipeline:
    1. Load data
    2. Extract features using sliding window
    3. Build graph based on feature distances
    4. Calculate Ricci curvature
    5. Visualize and analyze results
    """
    # Load the data
    data = load_data(file_path)
    if data is None:
        return None

    # Extract features
    print(f"Extracting features with window size {window_size} and step size {step_size}...")
    features, timestamps = extract_features_sliding_window(data, window_size, step_size)

    # Build graph
    print(f"Building graph using {distance_method} distance with epsilon={epsilon}...")
    if distance_method == 'cosine':
        G = build_graph_approximate_knn(features, timestamps, distance_method, k_neighbors=5)
    else:
        G = build_graph(features, timestamps, distance_method, epsilon)


    # Calculate Ricci curvature
    print("Calculating Ricci curvature...")
    G_ricci = calculate_ricci_curvature(G)

    if G_ricci and visualize:
        print("Visualizing graph...")
        visualize_graph(G_ricci, f"Time Series Graph ({distance_method}, ε={epsilon})")

    # Return results for further analysis if needed
    return {
        'graph': G_ricci,
        'features': features,
        'timestamps': timestamps
    }

if __name__ == "__main__":
    # Replace with your file path
    file_path = '/content/drive/MyDrive/Data/processed_stock.csv'

    # Check if file_path is a directory
    if os.path.isdir(file_path):
        # Find first data file in directory
        data_files = [f for f in os.listdir(file_path) if f.endswith(('.csv', '.xlsx', '.xls'))]
        if data_files:
            file_path = os.path.join(file_path, data_files[0])
            print(f"Found data file: {file_path}")
        else:
            print(f"No data files found in {file_path}")
            exit(1)

    # Example usage with all three distance methods
    for method in ['euclidean', 'cosine', 'dtw']:
        print(f"\n*** Analyzing with {method.upper()} distance ***")
        result = analyze_time_series(
            file_path=file_path,
            window_size=20,           # Size of feature window
            step_size=4,              # Step size for sliding window
            distance_method=method,   # Distance method
            epsilon=0.05,              # Threshold for edge creation
            visualize=True            # Whether to visualize results
        )

        if result and result['graph']:
            # Calculate average Ricci curvature
            ricci_values = [data['ricciCurvature'] for u, v, data in result['graph'].edges(data=True)]
            avg_ricci = sum(ricci_values) / len(ricci_values) if ricci_values else 0
            print(f"Average Ricci curvature: {avg_ricci:.4f}")
