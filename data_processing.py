import numpy as np
import csv
import re
import matplotlib.pyplot as plt


REAL_MODELS = [
    "ship",
    "goku",
    "blue kusudama",
    "crow",
    "whale spout",
    "cactus",
    "dog",
    "defect",
    "flat bull",
    "sheep",
    "tessellation",
    "rift scuttler",
    "charizard",
]
AI_MODELS = [
    "cute bunny",
    "ivysaur",
    "buff cat",
    "dragon",
    "simple whale",
    "triangle twists",
    "white bunny",
    "armored brute",
    "blue knight",
    "3d bull",
    "peppermint kusudama"
]
#partial credit for "Not Sure" responses
NOT_SURE = 0

def clean_text(text):
    """Remove newlines within cells and non-ASCII characters from text."""
    if not isinstance(text, str):
        return text
    # Remove newlines that are WITHIN the cell content
    # csv.reader already handled the row-delimiting newlines
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove non-ASCII characters (including emojis)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Clean up multiple spaces
    text = ' '.join(text.split())
    return text

def import_and_clean_csv(filepath):
    """
    Import CSV and clean it into a numpy array.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    numpy.ndarray
        Cleaned array with rows 3+ and columns 2-28 (0-indexed: cols 1-27)
    """
    rows = []
    
    # Read CSV with proper handling of quoted fields containing newlines
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        
        # Skip first two rows
        next(reader, None)
        next(reader, None)
        
        for row in reader:
            if len(row) >= 28:  # Ensure row has enough columns
                # Extract columns 2-28 (indices 2-27 in 0-indexed)
                selected_cols = row[2:28]
                # Clean each cell
                cleaned_row = [clean_text(cell) for cell in selected_cols]
                rows.append(cleaned_row)
    
    # Convert to numpy array
    return np.array(rows, dtype=str)

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "raw_data.csv"
    data = import_and_clean_csv(csv_file)

    data[:,0][data[:,0]=="I've folded Satoshi Kamiya's Ancient Dragon (or similar difficulty)"] = "advanced"
    data[:,0][data[:,0]=="Some experience with simple models"] = "intermediate"
    data[:,0][data[:,0]=="No experience"] = "beginner"
    
    num_real = len(REAL_MODELS)
    num_ai = len(AI_MODELS)
    num_total = num_real + num_ai

    data[:,1:num_real+1][data[:,1:num_real+1]=="Real"] = "1"
    data[:,1:num_real+1][data[:,1:num_real+1]=="AI"] = "0"
    data[data == "Not sure"] = NOT_SURE
    data[:,num_real+1:num_real+num_ai+1][data[:,num_real+1:num_real+num_ai+1]=="AI"] = "1"
    data[:,num_real+1:num_real+num_ai+1][data[:,num_real+1:num_real+num_ai+1]=="Real"] = "0"
    
    column_headers = ["experience"] + REAL_MODELS + AI_MODELS + ["confidence"]
    cleaned_data = np.vstack([np.array(column_headers), data])
    np.savetxt("cleaned_data.csv", cleaned_data, delimiter=",", fmt="%s")

    # =========================

    scores = np.sum(cleaned_data[1:, 1:num_real+num_ai+1].astype(float), axis=1)
    # lean = (data=="Real").sum(axis=1)/13 - (data=="AI").sum(axis=1)/11

    advanced_scores = scores[cleaned_data[1:, 0] == "advanced"]
    intermediate_scores = scores[cleaned_data[1:, 0] == "intermediate"]
    beginner_scores = scores[cleaned_data[1:, 0] == "beginner"]

    fig, axes = plt.subplots(3,1, figsize=(15, 10))

    axes[0].hist(advanced_scores, bins=range(0, 26), edgecolor='black', color='green')
    axes[0].set_title("Advanced")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 24)

    axes[1].hist(intermediate_scores, bins=range(0, 26), edgecolor='black',color = "blue")
    axes[1].set_title("Intermediate")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0, 24)

    axes[2].hist(beginner_scores, bins=range(0, 26), edgecolor='black',color = "red")
    axes[2].set_title("Beginner")
    axes[2].set_xlabel("Score")
    axes[2].set_ylabel("Frequency")
    axes[2].set_xlim(0, 24)

    plt.tight_layout()
    plt.show()


    