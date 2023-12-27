import tkinter as tk
from pandasgui import show
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Function to fetch dataset and display table
def fetch_and_display_table():
    # Fetch dataset
    automobile = fetch_ucirepo(id=10)

    # Data (as pandas dataframes)
    X = automobile.data.features
    y = automobile.data.targets

    # Combine features and targets
    df_table = pd.DataFrame(X)
    df_table['Target'] = y

    # Display the DataFrame in a GUI
    gui = show(df_table, title="Dataset Table")

    # Access the Grapher tab
    grapher_tab = gui.tabs['Grapher']

    # You can add code here to customize the Grapher tab, such as setting x and y variables.

# Start the GUI
root = tk.Tk()

# Add a button to fetch dataset and display table
button = tk.Button(root, text="Fetch and Display Table", command=fetch_and_display_table)
button.pack()

# Start the Tkinter main loop
root.mainloop()
