# Data_Mining_Assignment_Movie_Clustering

## Introduction

This repository contains an implementation of k-means clustering algorithm to analyze movie data from the IMDB database. The program enables users to group movies based on their IMDB ratings and detect outliers, providing insights into the distribution of ratings among different clusters.

## Problem Description
Consider a dataset scraped from IMDB's official website containing ratings of the top 2000 movies between 1921 and 2010. The task is to implement the k-means algorithm to cluster movies based on their similarity in IMDB ratings. Users can input the number of clusters (k) during runtime, and the program will detect outliers, if any, while clustering the movies.

## Features
1. **User-Friendly Interface:** The program provides a graphical user interface (GUI) for easy interaction.
2. **Variable Data Selection:** Users can specify the percentage of data from the input file to be analyzed.
3. **Number of Clusters (K):** Users can input the number of clusters for the k-means algorithm.
4. **Random Initial Centroids:** Initial centroids for clustering are chosen randomly.
5. **Outlier Detection:** The program detects outlier movies based on their IMDB ratings.
6. **Results Display:** The final output displays the clustered movies along with outlier records.

## Screenshots
### Main GUI

![Main GUI](https://github.com/Abdallah531/Data_Mining_Assignment_Movie_Clustering/assets/117390537/28573fa9-3a83-4df0-a00e-ac3a9310929a)

### Data Distribution

![Data Distribution](https://github.com/Abdallah531/Data_Mining_Assignment_Movie_Clustering/assets/117390537/8a839518-7971-4c79-a253-835d74d16568)


### Clustering Results

![Clustering Results](https://github.com/Abdallah531/Data_Mining_Assignment_Movie_Clustering/assets/117390537/13d48967-38da-4652-9d91-aba30103f038)

## Requirements
- Python 3.x
- pandas
- tkinter (for GUI)
- numpy
- matplotlib
- seaborn

## How to Use
1. **Input File:** Provide a CSV file containing movie data with IMDB ratings.
2. **Percentage of Data:** Specify the percentage of data to be analyzed.
3. **Number of Clusters (K):** Enter the desired number of clusters for k-means clustering.
4. Click the "Process Data" button to execute the clustering algorithm.
5. View the results in the text area provided, displaying clustered movies and outlier records.
