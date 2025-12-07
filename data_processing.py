import numpy as np
import csv
import re
from scipy.stats import binom
from numpy.polynomial import polynomial as P
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
NOT_SURE = 0.3

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

def extract_comments(filepath, output_file1 = "qualitative_responses1.txt", output_file2 = "qualitative_responses2.txt"):
    """
    Extract comments from the last column of the CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    list
        List of cleaned comments from the last column
    """
    comments1 = [] #"what features did you look for"
    comments2 = [] #"any other thoughts"
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        # Skip first two rows
        next(reader, None)
        next(reader, None)
        for row in reader:
            if len(row) >= 28:  # Ensure row has enough columns
                selected_cols = row[28:30]
                cleaned_cols = [clean_text(cell) for cell in selected_cols]
                if len(cleaned_cols[0]) > 4:
                    comments1.append(cleaned_cols[0])
                if len(cleaned_cols[1]) > 4:
                    comments2.append(cleaned_cols[1])
    


    with open(output_file1, 'w', encoding='utf-8') as f1:
        f1.write("The following were responses to the question, \"What features did you look for to help you classify what you saw?\" Responses are sorted in decreasing order of identification accuracy score.\n==========================================\n")
        for comment in comments1:
            f1.write(comment + '\n')
    with open(output_file2, 'w', encoding='utf-8') as f2:
        f2.write("The following were responses to the question, \"Any other thoughts about AI and origami?\" \n==========================================\n")
        for comment in comments2:
            f2.write(comment + '\n')
    return comments1, comments2
# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path

    extract_comments("raw_data.csv")
    

    csv_file = "raw_data.csv"
    responses = import_and_clean_csv(csv_file)

    responses[:,0][responses[:,0]=="I've folded Satoshi Kamiya's Ancient Dragon (or similar difficulty)"] = "advanced"
    responses[:,0][responses[:,0]=="Some experience with simple models"] = "intermediate"
    responses[:,0][responses[:,0]=="No experience"] = "beginner"
    
    num_real = len(REAL_MODELS)
    num_ai = len(AI_MODELS)
    num_total = num_real + num_ai

    scores = responses.copy()
    scores[:,1:num_real+1][scores[:,1:num_real+1]=="Real"] = "1"
    scores[:,1:num_real+1][scores[:,1:num_real+1]=="AI"] = "0"
    scores[scores == "Not sure"] = NOT_SURE
    scores[:,num_real+1:num_real+num_ai+1][scores[:,num_real+1:num_real+num_ai+1]=="AI"] = "1"
    scores[:,num_real+1:num_real+num_ai+1][scores[:,num_real+1:num_real+num_ai+1]=="Real"] = "0"
    
    column_headers = ["experience"] + REAL_MODELS + AI_MODELS + ["confidence"]
    np.savetxt("cleaned_data.csv", np.vstack([np.array(column_headers), scores]), delimiter=",", fmt="%s")


    scores = np.sum(scores[:, 1:num_real+num_ai+1].astype(float), axis=1)
    # lean = (data=="Real").sum(axis=1)/13 - (data=="AI").sum(axis=1)/11

    advanced_responses = responses[responses[:, 0] == "advanced"]
    intermediate_responses = responses[responses[:, 0] == "intermediate"]
    beginner_responses = responses[responses[:, 0] == "beginner"]

    advanced_scores = scores[responses[:, 0] == "advanced"]
    intermediate_scores = scores[responses[:, 0] == "intermediate"]
    beginner_scores = scores[responses[:, 0] == "beginner"]

    # =========================
    # Scoring Histograms
    # =========================

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Create binomial distribution for overlay
    x = np.arange(0, 25)
    binomial_pmf = binom.pmf(x, num_total, 0.5)
    
    for scores_data, ax, title, color in [
        (advanced_scores, axes[0], "Advanced", 'green'),
        (intermediate_scores, axes[1], "Intermediate", 'blue'),
        (beginner_scores, axes[2], "Beginner", 'red')
    ]:
        # Compute mean and 95% CI
        mean = np.mean(scores_data)
        se = np.std(scores_data, ddof=1) / np.sqrt(len(scores_data))
        ci = 1.96 * se
        
        # Histogram centered on tick marks
        ax.hist(scores_data, bins=np.arange(-0.5, 25.5), edgecolor='black', color=color, density=True, alpha=0.7)
        ax.plot(x, binomial_pmf, 'ko-', linewidth=2, markersize=6, label='Binomial(p=0.5)')
        
        # Plot mean with 95% CI
        ax.errorbar(mean, ax.get_ylim()[1] * 0.95, xerr=ci, fmt='o', color='black', capsize=5, markersize=8, label=f'Mean ± 95% CI: {mean:.2f} ± {ci:.2f}')
        
        # Vertical line at x=12
        ax.axvline(x=12, color='purple', linestyle='--', linewidth=2, label='x=12')
        
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 24)
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/score_histograms.png")

    # =========================
    # Per-question success rate
    # =========================


    
    # ========================
    # Confidence vs actual score
    # TODO: line of best fit isn't accurate. Expected is 50% for 0 confidence and 100% for max confidence. Let confidence x mean an x percent chance of getting it right and 1-x percent chance of coin tossing.
    # ========================
    fig, ax = plt.subplots(figsize=(12, 9))
    advanced_confidence = advanced_responses[:, -1].astype(float)
    intermediate_confidence = intermediate_responses[:, -1].astype(float)
    beginner_confidence = beginner_responses[:, -1].astype(float)
    ax.scatter(advanced_confidence, advanced_scores, color='green', alpha=0.05, label='Advanced')
    ax.scatter(intermediate_confidence, intermediate_scores, color='blue', alpha=0.05, label='Intermediate')
    ax.scatter(beginner_confidence, beginner_scores, color='red', alpha=0.05, label='Beginner')

    for confidence, scores, color in [
        (advanced_confidence, advanced_scores, 'green'),
        (intermediate_confidence, intermediate_scores, 'blue'),
        (beginner_confidence, beginner_scores, 'red')
    ]:
        # Fit a line to the data
        coeffs = np.polyfit(confidence, scores, 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(confidence.min(), confidence.max(), 100)
        ax.plot(x_line, poly(x_line), color=color, linewidth=2)
    for confidence, scores, color, label in [
        (advanced_confidence, advanced_scores, 'green', 'Advanced'),
        (intermediate_confidence, intermediate_scores, 'blue', 'Intermediate'),
        (beginner_confidence, beginner_scores, 'red', 'Beginner')
    ]:
        coeffs = np.polyfit(confidence, scores, 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(confidence.min(), confidence.max(), 100)
        ax.plot(x_line, poly(x_line), color=color, linewidth=2)
        
        # Calculate R-squared
        y_pred = poly(confidence)
        ss_res = np.sum((scores - y_pred) ** 2)
        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Print slope and R-squared
        slope = coeffs[0]
        ax.text(0.05, 0.95 - (0.1 * (['Advanced', 'Intermediate', 'Beginner'].index(label))), 
            f"{label}: slope={slope:.4f}, R²={r_squared:.4f}", 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.set_xlim(1, 10)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25))
    ax.set_xlabel("Self-confidence")
    ax.set_ylabel("Actual score")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/confidence_vs_score.png")