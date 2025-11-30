import pandas as pd
import matplotlib.pyplot as plt
import sys


def load_csv_columns(filepath, x_col_name, y_col_name):
    """
    Loads a CSV file and extracts two specified columns for X and Y axes.

    Args:
        filepath (str): The path to the CSV file.
        x_col_name (str): The exact name of the column for the x-axis.
        y_col_name (str): The exact name of the column for the y-axis.

    Returns:
        tuple: A tuple containing (x_data, y_data) as pandas Series.
               Returns (None, None) if an error occurs.
    """
    try:
        df = pd.read_csv(filepath)

        if x_col_name not in df.columns:
            print(f"Error: Column '{x_col_name}' not found in the CSV file.")
            print(f"Available columns are: {list(df.columns)}")
            return None, None

        if y_col_name not in df.columns:
            print(f"Error: Column '{y_col_name}' not found in the CSV file.")
            print(f"Available columns are: {list(df.columns)}")
            return None, None

        x_data = df[x_col_name]
        y_data = df[y_col_name]

        print(f"Successfully loaded columns '{x_col_name}' and '{y_col_name}'.")
        return x_data, y_data

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{filepath}' is empty.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


def plot_data(x_data, y_data, x_col_name, y_col_name, output_filename="plot.png"):
    """
    Generates and saves a scatter plot from the x and y data.

    Args:
        x_data (pd.Series): Data for the x-axis.
        y_data (pd.Series): Data for the y-axis.
        x_col_name (str): Name of the x-axis column for labeling.
        y_col_name (str): Name of the y-axis column for labeling.
        output_filename (str): The filename to save the plot as (e.g., "my_plot.png").
    """
    try:
        plt.figure(figsize=(10, 6))  # Set a reasonable figure size

        # Use a scatter plot, which is generally best for X-Y data
        plt.scatter(x_data, y_data, alpha=0.7, edgecolors='k')

        # Add labels and title
        plt.title(f'{y_col_name} vs. {x_col_name}', fontsize=16)
        plt.xlabel(x_col_name, fontsize=12)
        plt.ylabel(y_col_name, fontsize=12)

        # Add a grid for readability
        plt.grid(True)

        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig(output_filename)

        print(f"\n--- Plotting ---")
        print(f"Plot saved successfully as '{output_filename}'")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")


# --- Main execution block ---
if __name__ == "__main__":

    # 1. Get input from the user
    file_path = "harmonic0.csv"
    x_column = "x"
    y_column = "psi"

    print("\n--- Loading Data ---")
    # 2. Call the function to load the data
    x_axis, y_axis = load_csv_columns(file_path, x_column, y_column)

    # 3. Check if data was loaded successfully
    if x_axis is not None and y_axis is not None:

        # Print a sample
        print("\n--- Data Sample (first 5 rows) ---")
        print(f"{x_column} (X-Axis):")
        print(x_axis.head())
        print(f"\n{y_column} (Y-Axis):")
        print(y_axis.head())

        # 4. If data is valid, call the plot function
        plot_data(x_axis, y_axis, x_column, y_column)

    else:
        print("Data loading failed. Exiting.")