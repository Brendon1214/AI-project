import pandas as pd
import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import colors
import tkinter as tk
from tkinter import Button, Label, StringVar, ttk

import threading

# Global variables
is_paused = False
progress_var = None

# Function to scale data using Min-Max scaling
def minmax_scaler(data):
    # Separate numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns
    non_numeric_cols = list(set(data.columns) - set(numeric_cols))

    # Scale numeric columns
    numeric_data = data[numeric_cols]
    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(numeric_data)

    # Check for and replace NaN and infinite values
    scaled_numeric = np.nan_to_num(scaled_numeric)

    # One-hot encode non-numeric columns
    non_numeric_data = pd.get_dummies(data[non_numeric_cols])

    # Concatenate scaled numeric and one-hot encoded non-numeric data
    scaled_data = np.concatenate([scaled_numeric, non_numeric_data], axis=1)

    return scaled_data

# Function to compute Euclidean distance
def e_distance(x, y):
    if x.ndim > 1:
        x = x.flatten()
    if y.ndim > 1:
        y = y.flatten()
    return distance.euclidean(x, y)

# Function to compute Manhattan distance
def m_distance(x, y):
    return distance.cityblock(x, y)

# Function to find the winning neuron in SOM
def winning_neuron(data, t, som, num_rows, num_cols):
    winner = [0, 0]
    shortest_distance = np.sqrt(data.shape[1])
    input_data = data[t][:, np.newaxis] if data[t].ndim == 1 else data[t]
    for row in range(num_rows):
        for col in range(num_cols):
            dist = e_distance(som[row][col], input_data)
            if dist < shortest_distance:
                shortest_distance = dist
                winner = [row, col]
    return winner

# Function to decay learning rate and neighborhood range
def decay(step, max_steps, max_learning_rate, max_neighbourhood_range):
    coefficient = 1.0 - (np.float64(step) / max_steps)
    learning_rate = coefficient * max_learning_rate
    neighbourhood_range = ceil(coefficient * max_neighbourhood_range)
    return learning_rate, neighbourhood_range

# Function to train Self-Organizing Map (SOM)
def som_training(train_data, num_rows, num_cols, max_neighbourhood_range, max_learning_rate, max_steps, label_var,
                 train_y=None):
    num_dims = train_data.shape[1]
    np.random.seed(40)
    som = np.random.random_sample(size=(num_rows, num_cols, num_dims))

    for step in range(max_steps):
        if is_paused:
            continue  # Skip the iteration if paused

        progress_var.set((step + 1) / max_steps * 100)  # Update progress bar

        if (step + 1) % 1000 == 0:
            label_var.set(f"Iteration: {step + 1}")
            root.update()  # Update the Tkinter GUI

        learning_rate, neighbourhood_range = decay(step, max_steps, max_learning_rate, max_neighbourhood_range)

        t = np.random.randint(0, high=train_data.shape[0])
        winner = winning_neuron(train_data, t, som, num_rows, num_cols)

        for row in range(num_rows):
            for col in range(num_cols):
                if m_distance([row, col], winner) <= neighbourhood_range:
                    som[row][col] += learning_rate * (train_data[t] - som[row][col])

    print("SOM training completed")
    label_var.set("SOM training completed")

    # Label the SOM nodes
    label_map = label_som_nodes(train_data, train_y, num_rows, num_cols, som)
    return som, label_map

# Function to label SOM nodes
def label_som_nodes(train_data_norm, train_y, num_rows, num_cols, som):
    label_map = np.full((num_rows, num_cols), None, dtype=object)

    for t in range(train_data_norm.shape[0]):
        winner = winning_neuron(train_data_norm, t, som, num_rows, num_cols)
        row, col = winner
        label = train_y.iloc[t].item()  # Convert Series to a simple value

        if label_map[row, col] is None:
            label_map[row, col] = [label]
        else:
            label_map[row, col].append(label)

    for i in range(num_rows):
        for j in range(num_cols):
            if label_map[i, j] is not None:
                label_map[i, j] = max(set(label_map[i, j]), key=label_map[i, j].count)

    return label_map

# Function to evaluate accuracy on the test set
def evaluate_accuracy(test_data, test_y, som, label_map, num_rows, num_cols):
    winner_labels = []

    for t in range(test_data.shape[0]):
        input_data = test_data[t][:, np.newaxis] if test_data[t].ndim == 1 else test_data[t]
        winner = winning_neuron(input_data, t, som, num_rows, num_cols)
        row = winner[0]
        col = winner[1]
        predicted = label_map[row][col]

        # Check for None values and assign a default label if needed
        if predicted is None:
            predicted = 'default_label'

        winner_labels.append(predicted)

    # Convert winner_labels to match the format of test_y
    unique_labels_test = np.unique(test_y)
    label_dict_test = {label: idx for idx, label in enumerate(unique_labels_test)}
    winner_labels_numeric = np.array([label_dict_test[label] for label in winner_labels])

    accuracy = accuracy_score(test_y, winner_labels_numeric)
    return accuracy

# Function to display SOM Plot GUI
def display_som_plot_gui_func(label_map, max_steps):
    cmap = colors.ListedColormap(
        ['white', 'black', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown', 'pink'])

    # Convert labels to numeric values
    label_map_numeric = np.zeros_like(label_map, dtype=float)
    unique_labels = np.unique([label for row in label_map for label in row if label is not None])
    label_dict = {label: idx for idx, label in enumerate(unique_labels)}

    for i in range(label_map.shape[0]):
        for j in range(label_map.shape[1]):
            if label_map[i, j] is not None:
                label_map_numeric[i, j] = label_dict[label_map[i, j]]

    # Plot SOM
    plt.imshow(label_map_numeric, cmap=cmap)
    plt.colorbar(ticks=range(len(unique_labels)), label='Labels')
    plt.title(f'SOM after {max_steps} iterations')
    plt.show()

# Function to be called when the button is clicked
def on_button_click(label_var):
    global is_paused, progress_var

    # Reset global variables
    is_paused = False

    # Create a progress bar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.pack()

    # Create a thread for SOM training
    som_thread = threading.Thread(target=som_training_thread, args=(label_var,))
    som_thread.start()

# Function to pause SOM training
def on_pause_click():
    global is_paused
    is_paused = not is_paused

# Function to run SOM training in a separate thread
def som_training_thread(label_var):
    global som, label_map

    # Load dataset
    from ucimlrepo import fetch_ucirepo
    automobile = fetch_ucirepo(id=10)
    X = automobile.data.features
    y = automobile.data.targets

    # Split the dataset into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)  # check the shapes

    if train_y is None:
        print("Error: 'train_y' is None. Make sure the dataset is loaded correctly.")
        return

    # Hyperparameters for SOM
    num_rows = 10
    num_cols = 10
    max_neighbourhood_range = 4
    max_learning_rate = 0.5
    max_steps = int(7.5 * 10e3)

    # Train SOM
    train_data = pd.concat([train_x, train_y], axis=1)
    train_data_norm = minmax_scaler(train_data)
    som, label_map = som_training(train_data_norm, num_rows, num_cols, max_neighbourhood_range,
                                  max_learning_rate, max_steps, label_var, train_y)

    # Display SOM Plot GUI in the main thread
    display_som_plot_gui_func(label_map, max_steps)

    # Evaluate accuracy on the test set
    test_data_norm = minmax_scaler(test_x)
    accuracy = evaluate_accuracy(test_data_norm, test_y, som, label_map, num_rows, num_cols)

    # Display accuracy in the GUI
    label_var.set(f"SOM training completed\nAccuracy: {accuracy:.2%}")

# Start the GUI
def initialize_gui():
    global root
    root = tk.Tk()
    root.title("SOM Training GUI")

    # Add a button to start SOM training
    label_var = StringVar()
    label_var.set("Press the button to start SOM training")
    label = Label(root, textvariable=label_var)
    label.pack()

    button = Button(root, text="Start SOM Training", command=lambda: on_button_click(label_var))
    button.pack()

    pause_button = Button(root, text="Pause", command=on_pause_click)
    pause_button.pack()

    # Start the Tkinter main loop
    root.mainloop()

# Start the GUI
initialize_gui()
