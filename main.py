import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


def read_file_and_sample(file_path, percentage):
    df = pd.read_csv(file_path)  # Assuming CSV file
    sampled_df = df.sample(frac=percentage / 100)


    # Calculate Z-score for 'IMDB Rating' column
    z_scores = (sampled_df['IMDB Rating'] - sampled_df['IMDB Rating'].mean()) / sampled_df['IMDB Rating'].std()

    # Define a threshold for outliers (e.g., Z-score > 2 or Z-score < -2)
    threshold = 2

    # Detect outliers based on Z-score
    outliers = sampled_df[(z_scores > threshold) | (z_scores < -threshold)]

    # Draw Histogram Plot for z scores
    plt.figure(figsize=(10, 6))
    sns.histplot(data=z_scores, kde=True, color='blue')
    plt.title('Density Histogram Plot of z scores')
    plt.grid(True)
    plt.show()

    # # Print outliers
    print("Outliers Detected:")
    print(len(outliers[['Movie Name', 'IMDB Rating']]))

    # Remove outliers from the dataset
    sampled_df = sampled_df[(z_scores <= threshold) & (z_scores >= -threshold)]

    # Sample the remaining data

    return sampled_df[['Movie Name', 'IMDB Rating']]  # Extracting movie name and IMDB rating


def k_means_clustering(data, k, max_iterations=1000, tolerance=0):
    # Randomly select initial centroids
    centroids_indices = random.sample(range(len(data)), k)
    centroids = data.iloc[centroids_indices].values

    # Placeholder for clusters
    clusters = [[] for _ in range(k)]

    # Euclidean distance function
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Iterative process
    for _ in range(max_iterations):
        # Clear clusters for re-assignment
        clusters = [[] for _ in range(k)]

        # Assign each data point to the nearest centroid
        for index, row in data.iterrows():
            distances = [euclidean_distance(row.values[1:], centroid[1:]) for centroid in centroids]
            nearest_centroid_index = np.argmin(distances)
            clusters[nearest_centroid_index].append(row.values)

        # Update centroids
        old_centroids = centroids.copy()  # Keep track of old centroids for convergence check
        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                cluster_array = np.array(cluster)
                cluster_mean = np.mean(cluster_array[:, 1:], axis=0)
                centroids[i] = np.concatenate([[i], cluster_mean])

        # Check for convergence
        print(centroids)

        centroid_difference = np.sum(np.abs(centroids[:, 1:] - old_centroids[:, 1:]))
        if centroid_difference == tolerance:
            break

    # Calculate final centroids
    final_centroids = [centroid[1:] for centroid in centroids]
    return clusters, final_centroids


# Function to handle GUI
def browse_file():
    filename = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filename)


def process_data():
    file_path = entry.get()
    percentage = int(percentage_entry.get())
    k = int(k_entry.get())

    data = read_file_and_sample(file_path, percentage)
    clusters, final_centroids = k_means_clustering(data, k)

    # Display clusters on the GUI
    text_area.delete(1.0, tk.END)
    for i, cluster in enumerate(clusters):
        text_area.insert(tk.END, f"Cluster {i + 1}: \n")
        centroid = final_centroids[i]  # Extract centroid
        text_area.insert(tk.END, f"  Centroid: {centroid} \n")
        cluster_array = np.array(cluster)
        min_point = np.min(cluster_array[:, 1:], axis=0)
        max_point = np.max(cluster_array[:, 1:], axis=0)
        text_area.insert(tk.END, f"  Min Point: {min_point} \n")
        text_area.insert(tk.END, f"  Max Point: {max_point} \n")
        text_area.insert(tk.END, f"  Number of rows: {len(cluster_array)} \n")

        for point in cluster:
            text_area.insert(tk.END, f"  {point[0]} - IMDB Rating: {point[1]}")
            text_area.insert(tk.END, "\n")


# Create GUI
root = tk.Tk()
root.title("Movie Clustering")

style = ttk.Style()
style.configure('TLabel', font=('Arial', 12))
style.configure('TButton', font=('Arial', 12))

label = ttk.Label(root, text="Select File:")
label.grid(row=0, column=0, padx=10, pady=10)

entry = ttk.Entry(root, width=50)
entry.grid(row=0, column=1, padx=10, pady=10)

browse_button = ttk.Button(root, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2, padx=10, pady=10)

percentage_label = ttk.Label(root, text="Percentage of Data to Read (%):")
percentage_label.grid(row=1, column=0, padx=10, pady=10)

percentage_entry = ttk.Entry(root)
percentage_entry.grid(row=1, column=1, padx=10, pady=10)

k_label = ttk.Label(root, text="Number of Clusters (K):")
k_label.grid(row=2, column=0, padx=10, pady=10)

k_entry = ttk.Entry(root)
k_entry.grid(row=2, column=1, padx=10, pady=10)

process_button = ttk.Button(root, text="Process Data", command=process_data)
process_button.grid(row=3, columnspan=3, padx=10, pady=10)

# Add a scrolled text widget to display clustering results
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=25)
text_area.grid(row=6, columnspan=5, padx=10, pady=10)

root.mainloop()
