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


def plot_score_histograms(responses, scores, advanced_scores, intermediate_scores, beginner_scores):
    """
    Plot histograms of scores for each experience level with binomial overlay.
    
    Parameters:
    -----------
    responses : numpy.ndarray
        Array of survey responses
    scores : numpy.ndarray
        Array of computed scores
    advanced_scores : numpy.ndarray
        Scores for advanced users
    intermediate_scores : numpy.ndarray
        Scores for intermediate users
    beginner_scores : numpy.ndarray
        Scores for beginner users
    """
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

def create_stacked_bar_chart(responses, title):
    """
    Create a 100% stacked horizontal bar chart for model classifications.
    
    Parameters:
    -----------
    responses : np.ndarray
        Response data where column 0 is experience level and columns 1-24 are model responses
    title : str
        Title for the chart
    """
    num_real = len(REAL_MODELS)
    num_ai = len(AI_MODELS)
    all_models = REAL_MODELS + AI_MODELS
    
    # Calculate proportions for each model
    model_data = []
    
    for i, model_name in enumerate(all_models):
        col_idx = i + 1  # +1 because column 0 is experience level
        column = responses[:, col_idx]
        
        total = len(column)
        if total == 0:
            continue
            
        is_real_model = i < num_real
        
        if is_real_model:
            # For real models: correct="Real", uncertain="Not sure", incorrect="AI"
            correct = np.sum(column == "Real") / total
            uncertain = np.sum(column == "Not sure") / total
            incorrect = np.sum(column == "AI") / total
            color_correct = '#2ecc71'  # green
            color_uncertain = '#a9dfbf'  # desaturated green
        else:
            # For AI models: correct="AI", uncertain="Not sure", incorrect="Real"
            correct = np.sum(column == "AI") / total
            uncertain = np.sum(column == "Not sure") / total
            incorrect = np.sum(column == "Real") / total
            color_correct = '#e74c3c'  # red
            color_uncertain = '#f5b7b1'  # desaturated red
        
        model_data.append({
            'name': model_name,
            'correct': correct,
            'uncertain': uncertain,
            'incorrect': incorrect,
            'color_correct': color_correct,
            'color_uncertain': color_uncertain,
            'is_real': is_real_model
        })
    
    # Sort by correct percentage (descending)
    model_data.sort(key=lambda x: x['correct'], reverse=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(model_data) * 0.4 + 1.5))
    
    y_positions = np.arange(len(model_data))
    
    # Plot stacked bars (reversed order: correct on left, incorrect on right)
    for i, data in enumerate(model_data):
        # Correct (saturated color) - leftmost
        ax.barh(i, data['correct'], left=0,
                color=data['color_correct'], edgecolor='white', linewidth=0.5,
                label='_nolegend_')
        
        # Uncertain (desaturated color) - middle
        ax.barh(i, data['uncertain'], left=data['correct'],
                color=data['color_uncertain'], edgecolor='white', linewidth=0.5,
                label='_nolegend_')
        
        # Incorrect (grey) - rightmost
        ax.barh(i, data['incorrect'], left=data['correct'] + data['uncertain'], 
                color='#95a5a6', edgecolor='white', linewidth=0.5,
                label='_nolegend_')
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([d['name'] for d in model_data])
    ax.set_xlabel('Proportion of Responses', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for i, data in enumerate(model_data):
        # Only show percentage if segment is large enough
        if data['correct'] > 0.08:
            ax.text(data['correct']/2, i,
                   f"{data['correct']*100:.0f}%", 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white')
    
    plt.tight_layout()
    return fig


def calculate_skew(responses):
    """
    Calculate skew for each row in responses.
    
    Skew = (real + notsure/2)/24 - 13/24
    
    Parameters:
    -----------
    responses : np.ndarray
        Response data where column 0 is experience level and columns 1-24 are model responses
    
    Returns:
    --------
    np.ndarray
        Array of skew values, one per row
    """
    # Extract only the model response columns (1-24)
    model_responses = responses[:, 1:25]
    
    # Count "Real", "AI", and "Not sure" for each row
    real_counts = (model_responses == "Real").sum(axis=1)
    ai_counts = (model_responses == "AI").sum(axis=1)
    notsure_counts = (model_responses == "Not sure").sum(axis=1)
    
    # Calculate skew: (real + notsure/2)/24 - 13/24
    skew = (real_counts + notsure_counts/2) / 24 - 13/24
    
    return skew

def plot_skew_histograms(advanced_responses, intermediate_responses, beginner_responses):
    """
    Create histograms showing the distribution of skew for each experience level.
    
    Parameters:
    -----------
    advanced_responses, intermediate_responses, beginner_responses : np.ndarray
        Response data arrays for each experience level
    """
    # Calculate skew for each group
    advanced_skew = calculate_skew(advanced_responses)
    intermediate_skew = calculate_skew(intermediate_responses)
    beginner_skew = calculate_skew(beginner_responses)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Define consistent bins across all histograms for easy comparison
    bins = np.linspace(-0.6, 0.6, 30)
    
    # Plot each histogram
    datasets = [
        (advanced_skew, "Advanced participants", "#3ce74a", axes[0]),
        (intermediate_skew, "Intermediate participants", "#1282f3", axes[1]),
        (beginner_skew, "Beginner participants", "#db3734", axes[2])
    ]
    
    for skew_data, title, color, ax in datasets:
        ax.hist(skew_data, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical line at 0 (no skew)
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='No skew')
        
        # Add vertical line at mean
        mean_skew = np.mean(skew_data)
        ax.axvline(mean_skew, color='darkred', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_skew:.3f}')
        
        # Formatting
        ax.set_xlabel('Skew (toward AI ← 0 → toward Real)', fontsize=11)
        ax.set_ylabel('Number of Responses', fontsize=11)
        ax.set_title(f'{title} (n={len(skew_data)})', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        
        # Add statistics text box
        stats_text = f'Mean: {mean_skew:.3f}\nStd: {np.std(skew_data):.3f}\nMedian: {np.median(skew_data):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Distribution of Response Skew by Experience Level', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def extract_confidence(responses):
    """
    Extract self-confidence ratings from the last column.
    
    Parameters:
    -----------
    responses : np.ndarray
        Response data where the last column contains confidence ratings
    
    Returns:
    --------
    np.ndarray
        Array of confidence values, converted to float
    """
    # Extract last column and convert to float
    confidence = responses[:, -1].astype(float)
    return confidence

def plot_confidence_histograms(advanced_responses, intermediate_responses, beginner_responses):
    """
    Create histograms showing the distribution of self-confidence for each experience level.
    
    Parameters:
    -----------
    advanced_responses, intermediate_responses, beginner_responses : np.ndarray
        Response data arrays for each experience level
    """
    # Extract confidence for each group
    advanced_conf = extract_confidence(advanced_responses)
    intermediate_conf = extract_confidence(intermediate_responses)
    beginner_conf = extract_confidence(beginner_responses)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Define consistent bins across all histograms for easy comparison
    # Assuming confidence is on a scale (e.g., 1-5 or 1-10)
    min_val = min(advanced_conf.min(), intermediate_conf.min(), beginner_conf.min())
    max_val = max(advanced_conf.max(), intermediate_conf.max(), beginner_conf.max())
    bins = np.linspace(min_val, max_val, 20)
    
    # Plot each histogram
    datasets = [
        (advanced_conf, "Advanced Participants", "#61e73c", axes[0]),
        (intermediate_conf, "Intermediate Participants", "#12a1f3", axes[1]),
        (beginner_conf, "Beginner Participants", "#db3934", axes[2])
    ]
    
    for conf_data, title, color, ax in datasets:
        ax.hist(conf_data, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical line at mean
        mean_conf = np.mean(conf_data)
        ax.axvline(mean_conf, color='darkred', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_conf:.2f}')
        
        # Formatting
        ax.set_xlabel('Self-Confidence Rating', fontsize=11)
        ax.set_ylabel('Number of Responses', fontsize=11)
        ax.set_title(f'{title} (n={len(conf_data)})', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        
        # Add statistics text box
        stats_text = f'Mean: {mean_conf:.2f}\nStd: {np.std(conf_data):.2f}\nMedian: {np.median(conf_data):.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Distribution of Self-Confidence by Experience Level', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
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
    
    
    # =======================
    # extract comments
    # ======================
    extract_comments("raw_data.csv")

    # =========================
    # Scoring Histograms
    # =========================
    
    plot_score_histograms(responses, scores, advanced_scores, intermediate_scores, beginner_scores)

    # =========================
    # Per-question success rate
    # =========================

    fig1 = create_stacked_bar_chart(advanced_responses, 
                                     "Model-specific success rate: advanced participants")
    fig2 = create_stacked_bar_chart(intermediate_responses, 
                                     "Model-specific success rate: intermediate participants")
    fig3 = create_stacked_bar_chart(beginner_responses, 
                                     "Model-specific success rate: beginner participants")
    
    fig1.savefig('plots/advanced_classification.png', dpi=300, bbox_inches='tight')
    fig2.savefig('plots/intermediate_classification.png', dpi=300, bbox_inches='tight')
    fig3.savefig('plots/beginner_classification.png', dpi=300, bbox_inches='tight')
    
    # ========================
    # Response bias (skew/lean)
    # ========================
    fig = plot_skew_histograms(advanced_responses, intermediate_responses, beginner_responses)
    
    # Save or show
    fig.savefig('plots/skew_histograms.png', dpi=300, bbox_inches='tight')
    
    # ========================
    # Confidence vs actual score
    # TODO: line of best fit isn't accurate. Expected is 50% for 0 confidence and 100% for max confidence. Let confidence x mean an x percent chance of getting it right and 1-x percent chance of coin tossing.
    # ========================
    
    fig = plot_confidence_histograms(advanced_responses, intermediate_responses, beginner_responses)
    fig.savefig('plots/confidence_histograms.png', dpi=300, bbox_inches='tight')
    
    # fig, ax = plt.subplots(figsize=(12, 9))
    # advanced_confidence = advanced_responses[:, -1].astype(float)
    # intermediate_confidence = intermediate_responses[:, -1].astype(float)
    # beginner_confidence = beginner_responses[:, -1].astype(float)
    # ax.scatter(advanced_confidence, advanced_scores, color='green', alpha=0.05, label='Advanced')
    # ax.scatter(intermediate_confidence, intermediate_scores, color='blue', alpha=0.05, label='Intermediate')
    # ax.scatter(beginner_confidence, beginner_scores, color='red', alpha=0.05, label='Beginner')

    # for confidence, scores, color in [
    #     (advanced_confidence, advanced_scores, 'green'),
    #     (intermediate_confidence, intermediate_scores, 'blue'),
    #     (beginner_confidence, beginner_scores, 'red')
    # ]:
    #     # Fit a line to the data
    #     coeffs = np.polyfit(confidence, scores, 1)
    #     poly = np.poly1d(coeffs)
    #     x_line = np.linspace(confidence.min(), confidence.max(), 100)
    #     ax.plot(x_line, poly(x_line), color=color, linewidth=2)
    # for confidence, scores, color, label in [
    #     (advanced_confidence, advanced_scores, 'green', 'Advanced'),
    #     (intermediate_confidence, intermediate_scores, 'blue', 'Intermediate'),
    #     (beginner_confidence, beginner_scores, 'red', 'Beginner')
    # ]:
    #     coeffs = np.polyfit(confidence, scores, 1)
    #     poly = np.poly1d(coeffs)
    #     x_line = np.linspace(confidence.min(), confidence.max(), 100)
    #     ax.plot(x_line, poly(x_line), color=color, linewidth=2)
        
    #     # Calculate R-squared
    #     y_pred = poly(confidence)
    #     ss_res = np.sum((scores - y_pred) ** 2)
    #     ss_tot = np.sum((scores - np.mean(scores)) ** 2)
    #     r_squared = 1 - (ss_res / ss_tot)
        
    #     # Print slope and R-squared
    #     slope = coeffs[0]
    #     ax.text(0.05, 0.95 - (0.1 * (['Advanced', 'Intermediate', 'Beginner'].index(label))), 
    #         f"{label}: slope={slope:.4f}, R²={r_squared:.4f}", 
    #         transform=ax.transAxes, fontsize=10, verticalalignment='top')
    # ax.set_xlim(1, 10)
    # ax.set_xticks(range(1, 11))
    # ax.set_ylim(0, 24)
    # ax.set_yticks(range(0, 25))
    # ax.set_xlabel("Self-confidence")
    # ax.set_ylabel("Actual score")
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig("plots/confidence_vs_score.png")