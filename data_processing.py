import numpy as np
import csv
import re
from scipy.stats import binom
import matplotlib.pyplot as plt

# ==================== CONSTANTS ====================

REAL_MODELS = [
    "ship", "goku", "blue kusudama", "crow", "whale spout", "cactus",
    "dog", "defect", "flat bull", "sheep", "tessellation", 
    "rift scuttler", "charizard",
]
AI_MODELS = [
    "cute bunny", "ivysaur", "buff cat", "dragon", "simple whale",
    "triangle twists", "white bunny", "armored brute", "blue knight",
    "3d bull", "peppermint kus."
]

# Partial credit for "Not Sure" responses
NOT_SURE = 0.5

# Consistent color scheme for all plots
COLORS = {
    'advanced': '#2ecc71',      # Green
    'intermediate': '#3498db',   # Blue
    'beginner': '#e74c3c',       # Red
    'null_hypothesis': '#2c3e50' # Dark gray for binomial curves
}

# Plot styling constants
PLOT_STYLE = {
    'histogram_alpha': 0.8,
    'histogram_edgecolor': 'black',
    'histogram_linewidth': 1,
    'mean_line_width': 2.5,
    'mean_line_style': '-',
    'null_line_width': 3.0,
    'null_line_style': '--',
    'grid_alpha': 0.3,
    'errorbar_capsize': 5,
    'errorbar_markersize': 8
}

# ==================== DATA LOADING ====================

def clean_text(text):
    """Remove newlines within cells and non-ASCII characters from text."""
    if not isinstance(text, str):
        return text
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ' '.join(text.split())
    return text

def import_and_clean_csv(filepath):
    """Import CSV and clean it into a numpy array."""
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        
        for row in reader:
            if len(row) >= 28:
                selected_cols = row[2:28]
                cleaned_row = [clean_text(cell) for cell in selected_cols]
                rows.append(cleaned_row)
    
    return np.array(rows, dtype=str)

def extract_comments(filepath, output_file1="qualitative_responses1.txt", 
                    output_file2="qualitative_responses2.txt"):
    """Extract comments from the last two columns of the CSV file."""
    comments1, comments2 = [], []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        
        for row in reader:
            if len(row) >= 28:
                selected_cols = row[28:30]
                cleaned_cols = [clean_text(cell) for cell in selected_cols]
                if len(cleaned_cols[0]) > 4:
                    comments1.append(cleaned_cols[0])
                if len(cleaned_cols[1]) > 4:
                    comments2.append(cleaned_cols[1])
    
    with open(output_file1, 'w', encoding='utf-8') as f1:
        f1.write("The following were responses to the question, \"What features did you look for to help you classify what you saw?\" Responses are sorted in decreasing order of identification accuracy score.\n" + "="*42 + "\n")
        for comment in comments1:
            f1.write(comment + '\n')
    
    with open(output_file2, 'w', encoding='utf-8') as f2:
        f2.write("The following were responses to the question, \"Any other thoughts about AI and origami?\" \n" + "="*42 + "\n")
        for comment in comments2:
            f2.write(comment + '\n')
    
    return comments1, comments2

# ==================== DATA PROCESSING ====================

def split_by_experience(responses, scores=None):
    """Split responses (and optionally scores) by experience level."""
    levels = {
        'advanced': responses[responses[:, 0] == "advanced"],
        'intermediate': responses[responses[:, 0] == "intermediate"],
        'beginner': responses[responses[:, 0] == "beginner"]
    }
    
    if scores is not None:
        score_levels = {
            'advanced': scores[responses[:, 0] == "advanced"],
            'intermediate': scores[responses[:, 0] == "intermediate"],
            'beginner': scores[responses[:, 0] == "beginner"]
        }
        return levels, score_levels
    
    return levels

def calculate_skew(responses):
    """
    Calculate skew relative to the correct balance (13 Real, 11 AI):
    -1 = maximum AI overclassification
    0 = correct balance (13 Real, 11 AI)
    +1 = maximum Real overclassification
    """
    model_responses = responses[:, 1:25]
    real_counts = (model_responses == "Real").sum(axis=1)
    notsure_counts = (model_responses == "Not sure").sum(axis=1)
    
    # Effective Real count (Not sure counts as 0.5)
    effective_real = real_counts + notsure_counts / 2
    
    # Perfect balance would be 13 Real responses
    deviation = effective_real - 13
    
    # Normalize: max overclassification as Real is 11 (24-13)
    #           max overclassification as AI is -13 (0-13)
    skew = np.where(deviation >= 0,
                    deviation / 11,   # Real overclassification
                    deviation / 13)   # AI overclassification
    
    return skew

def extract_confidence(responses):
    """Extract self-confidence ratings from the last column."""
    return responses[:, -1].astype(float)

# ==================== PLOTTING HELPERS ====================

def setup_histogram_axis(ax, xlabel, ylabel, title, xlim=None, ylim=None):
    """Apply consistent formatting to histogram axes."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=PLOT_STYLE['grid_alpha'], linestyle='--')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

def add_statistics_box(ax, data, x_pos=0.02, y_pos=0.98):
    """Add a statistics text box to the plot."""
    stats_text = f'Mean: {np.mean(data):.3f}\nStd: {np.std(data, ddof=1):.3f}\nMedian: {np.median(data):.3f}'
    ax.text(x_pos, y_pos, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

def plot_histogram_with_mean(ax, data, bins, color, label):
    """Plot histogram with mean line and return mean value."""
    ax.hist(data, bins=bins, color=color, alpha=PLOT_STYLE['histogram_alpha'], 
            edgecolor=PLOT_STYLE['histogram_edgecolor'], 
            linewidth=PLOT_STYLE['histogram_linewidth'], 
            label='_nolegend_')
    
    mean_val = np.mean(data)
    ax.axvline(mean_val, color='darkred', 
              linestyle=PLOT_STYLE['mean_line_style'], 
              linewidth=PLOT_STYLE['mean_line_width'], 
              label=f'Mean: {mean_val:.3f}')
    
    return mean_val

# ==================== MAIN PLOTTING FUNCTIONS ====================
def plot_score_histograms(responses, scores, score_levels):
    """Plot histograms of accuracy for each experience level with binomial overlay."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    
    num_total = len(REAL_MODELS) + len(AI_MODELS)
    
    # Convert scores to accuracy percentage
    x_score = np.arange(0, num_total + 1)
    x_accuracy = (x_score / num_total) * 100  # Convert to percentage
    binomial_pmf = binom.pmf(x_score, num_total, 0.5)
    
    # Bins in percentage space (centered on each possible score converted to %)
    bins_score = np.arange(-0.5, num_total + 0.5, 1)
    bins_accuracy = (bins_score / num_total) * 100
    
    for level, ax in zip(['advanced', 'intermediate', 'beginner'], axes):
        scores_data = score_levels[level]
        accuracy_data = (scores_data / num_total) * 100  # Convert to percentage
        color = COLORS[level]
        
        # Histogram (density normalized)
        ax.hist(accuracy_data, bins=bins_accuracy, edgecolor=PLOT_STYLE['histogram_edgecolor'], 
                color=color, density=True, alpha=PLOT_STYLE['histogram_alpha'],
                linewidth=PLOT_STYLE['histogram_linewidth'])
        
        # Binomial overlay (scale density to match percentage bins)
        # Need to scale the PMF by (num_total / 100) to account for bin width change
        binomial_pmf_scaled = binomial_pmf * (num_total / 100)
        ax.plot(x_accuracy, binomial_pmf_scaled, color=COLORS['null_hypothesis'], 
                linestyle=PLOT_STYLE['null_line_style'], 
                linewidth=PLOT_STYLE['null_line_width'], 
                marker='.', markersize=4, label='Coin toss binomial')
        
        # Vertical line at chance (50%)
        ax.axvline(x=50, color=COLORS['null_hypothesis'], 
                  linestyle=PLOT_STYLE['null_line_style'], 
                  linewidth=PLOT_STYLE['null_line_width'])
        
        # Mean with 95% CI (converted to percentage)
        mean = np.mean(accuracy_data)
        se = np.std(accuracy_data, ddof=1) / np.sqrt(len(accuracy_data))
        ci = 1.96 * se
        
        # Mean line and CI
        ax.axvline(mean, color='darkred', 
                  linestyle=PLOT_STYLE['mean_line_style'], 
                  linewidth=PLOT_STYLE['mean_line_width'], 
                  label=f'Mean: {mean:.1f}%')
        ax.axvspan(mean - ci, mean + ci, 
                  alpha=0.2, color='darkred', label=f'95% CI: ±{ci:.1f}%')
        
        setup_histogram_axis(ax, "Accuracy (%)", "Density", 
                           f"{level.capitalize()} (n={len(accuracy_data)})",
                           xlim=(0, 100))
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    return fig

def create_stacked_bar_chart(responses, title):
    """Create a 100% stacked horizontal bar chart for model classifications."""
    num_real = len(REAL_MODELS)
    num_ai = len(AI_MODELS)
    all_models = REAL_MODELS + AI_MODELS
    
    model_data = []
    
    for i, model_name in enumerate(all_models):
        col_idx = i + 1
        column = responses[:, col_idx]
        total = len(column)
        
        if total == 0:
            continue
            
        is_real_model = i < num_real
        
        if is_real_model:
            correct = np.sum(column == "Real") / total
            uncertain = np.sum(column == "Not sure") / total
            incorrect = np.sum(column == "AI") / total
            color_correct = '#2ecc71'
            color_uncertain = '#a9dfbf'
        else:
            correct = np.sum(column == "AI") / total
            uncertain = np.sum(column == "Not sure") / total
            incorrect = np.sum(column == "Real") / total
            color_correct = '#e74c3c'
            color_uncertain = '#f5b7b1'
        
        model_data.append({
            'name': model_name,
            'correct': correct,
            'uncertain': uncertain,
            'incorrect': incorrect,
            'color_correct': color_correct,
            'color_uncertain': color_uncertain,
            'is_real': is_real_model
        })
    
    model_data.sort(key=lambda x: x['correct'], reverse=False)
    
    fig, ax = plt.subplots(figsize=(10, len(model_data) * 0.45 + 1.5))
    y_positions = np.arange(len(model_data))
    
    for i, data in enumerate(model_data):
        ax.barh(i, data['correct'], left=0,
                color=data['color_correct'], edgecolor='white', linewidth=0.7)
        ax.barh(i, data['uncertain'], left=data['correct'],
                color=data['color_uncertain'], edgecolor='white', linewidth=0.7)
        ax.barh(i, data['incorrect'], left=data['correct'] + data['uncertain'], 
                color='#95a5a6', edgecolor='white', linewidth=0.7)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([d['name'] for d in model_data], fontsize=16, 
                       rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Proportion of Responses', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=PLOT_STYLE['grid_alpha'], linestyle='--')
    
    for i, data in enumerate(model_data):
        if data['correct'] > 0.08:
            ax.text(data['correct']/2, i,
                   f"{data['correct']*100:.0f}%", 
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color='white')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='white', label='Correct (Real)'),
        Patch(facecolor='#e74c3c', edgecolor='white', label='Correct (AI)'),
        Patch(facecolor='#a9dfbf', edgecolor='white', label='Not Sure (Real)'),
        Patch(facecolor='#f5b7b1', edgecolor='white', label='Not Sure (AI)'),
        Patch(facecolor='#95a5a6', edgecolor='white', label='Incorrect')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=16)
    
    plt.tight_layout()
    return fig


def plot_skew_histograms(response_levels):
    """Create histograms showing the distribution of skew for each experience level."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    bins = np.linspace(-1, 1, 24)  # 40 bins for smoother histogram
    
    for level, ax in zip(['advanced', 'intermediate', 'beginner'], axes):
        skew_data = calculate_skew(response_levels[level])
        color = COLORS[level]
        
        # Plot histogram
        ax.hist(skew_data, bins=bins, color=color, alpha=PLOT_STYLE['histogram_alpha'], 
                edgecolor=PLOT_STYLE['histogram_edgecolor'], 
                linewidth=PLOT_STYLE['histogram_linewidth'])
        
        # Add zero line (no bias)
        ax.axvline(0, color=COLORS['null_hypothesis'], 
                  linestyle=PLOT_STYLE['null_line_style'], 
                  linewidth=PLOT_STYLE['null_line_width'], 
                  alpha=0.7, label='No bias')
        
        # Calculate mean and 95% CI
        mean_val = np.mean(skew_data)
        se = np.std(skew_data, ddof=1) / np.sqrt(len(skew_data))
        ci_width = 1.96 * se
        
        # Plot mean line
        ax.axvline(mean_val, color='darkred', 
                  linestyle=PLOT_STYLE['mean_line_style'], 
                  linewidth=PLOT_STYLE['mean_line_width'], 
                  label=f'Mean: {mean_val:.3f}')
        
        # Plot 95% CI as transparent region
        ax.axvspan(mean_val - ci_width, mean_val + ci_width, 
                  alpha=0.2, color='darkred', label=f'95% CI: ±{ci_width:.3f}')
        
        setup_histogram_axis(ax, 
            'Overclassification bias (toward AI ← 0 → toward Real)', 
            'Number of Responses',
            f'{level.capitalize()} (n={len(skew_data)})',
            xlim=(-1, 1))
        
        ax.legend(loc='upper left', fontsize=10)
    
    plt.suptitle('Distribution of Response Bias by Experience Level', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def plot_confidence_histograms(response_levels):
    """Create histograms showing the distribution of self-confidence for each experience level."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    
    # Bins from 0.5 to 10.5 so bars are centered on integers 1-10
    bins = np.arange(0.5, 11.5, 1)
    
    for level, ax in zip(['advanced', 'intermediate', 'beginner'], axes):
        conf_data = extract_confidence(response_levels[level])
        color = COLORS[level]
        
        # Plot histogram with no gaps
        ax.hist(conf_data, bins=bins, color=color, alpha=PLOT_STYLE['histogram_alpha'], 
                edgecolor=PLOT_STYLE['histogram_edgecolor'], 
                linewidth=PLOT_STYLE['histogram_linewidth'])
        
        # Calculate mean and 95% CI
        mean_val = np.mean(conf_data)
        se = np.std(conf_data, ddof=1) / np.sqrt(len(conf_data))
        ci_width = 1.96 * se
        
        # Plot mean line
        y_max = ax.get_ylim()[1]
        ax.axvline(mean_val, color='darkred', 
                  linestyle=PLOT_STYLE['mean_line_style'], 
                  linewidth=PLOT_STYLE['mean_line_width'], 
                  label=f'Mean: {mean_val:.2f}')
        
        # Plot 95% CI as transparent error bar (horizontal span)
        ax.axvspan(mean_val - ci_width, mean_val + ci_width, 
                  alpha=0.2, color='darkred', label=f'95% CI: ±{ci_width:.2f}')
        
        # Set x-axis with tick marks at center of each bar (1-10)
        ax.set_xticks(np.arange(1, 11))
        ax.set_xlim(0.5, 10.5)
        
        setup_histogram_axis(ax, 
            'Self-Confidence Rating (1 = least confident, 10 = most confident)', 
            'Number of Responses',
            f'{level.capitalize()} (n={len(conf_data)})')
        
        ax.legend(loc='upper right', fontsize=10)
        # add_statistics_box(ax, conf_data, x_pos=0.75, y_pos=0.98)
    
    plt.suptitle('Distribution of Self-Confidence by Experience Level', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_stacked_confidence_histogram(response_levels, COLORS):
    """Create a stacked histogram showing confidence distributions for all experience levels."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract confidence data for each level
    advanced_conf = response_levels['advanced'][:, -1].astype(float)
    intermediate_conf = response_levels['intermediate'][:, -1].astype(float)
    beginner_conf = response_levels['beginner'][:, -1].astype(float)
    
    # Bins from 0.5 to 10.5 so bars are centered on integers 1-10
    bins = np.arange(0.5, 11.5, 1)
    
    # Create histogram data for each group
    advanced_counts, _ = np.histogram(advanced_conf, bins=bins)
    intermediate_counts, _ = np.histogram(intermediate_conf, bins=bins)
    beginner_counts, _ = np.histogram(beginner_conf, bins=bins)
    
    # X positions (centers of bars)
    x_pos = np.arange(1, 11)
    bar_width = 0.9
    
    # Create stacked bars
    ax.bar(x_pos, beginner_counts, bar_width, 
           label=f'Beginner (n={len(beginner_conf)})', 
           color=COLORS['beginner'], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.bar(x_pos, intermediate_counts, bar_width, 
           bottom=beginner_counts,
           label=f'Intermediate (n={len(intermediate_conf)})', 
           color=COLORS['intermediate'], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.bar(x_pos, advanced_counts, bar_width, 
           bottom=beginner_counts + intermediate_counts,
           label=f'Advanced (n={len(advanced_conf)})', 
           color=COLORS['advanced'], alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Plot means with 95% CI for each group
    for level, conf_data, color, y_offset in [
        ('Beginner', beginner_conf, COLORS['beginner'], 0),
        ('Intermediate', intermediate_conf, COLORS['intermediate'], 1),
        ('Advanced', advanced_conf, COLORS['advanced'], 2)
    ]:
        mean_val = np.mean(conf_data)
        se = np.std(conf_data, ddof=1) / np.sqrt(len(conf_data))
        ci_width = 1.96 * se
        
        # Mark mean with a vertical line
        y_max = ax.get_ylim()[1]
        line_style = ['--', '--', '--'][y_offset]
        ax.axvline(mean_val, color=color, linestyle=line_style, 
                  linewidth=5, alpha=1,
                  label=f'{level} mean: {mean_val:.2f}')
    
    # Formatting
    ax.set_xlabel('Self-Confidence Rating (1 = least confident, 10 = most confident)', 
                 fontsize=12)
    ax.set_ylabel('Number of Responses (Stacked)', fontsize=12)
    ax.set_title('Combined Distribution of Self-Confidence by Experience Level', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 400)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    
    plt.tight_layout()
    return fig


def plot_confidence_vs_accuracy(response_levels, score_levels):
    """Create scatter plots of confidence vs accuracy for each experience level."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    
    num_total = len(REAL_MODELS) + len(AI_MODELS)
    
    # Null hypothesis line: (1, 50%) to (10, 100%)
    # This assumes no confidence = random guessing (50%), full confidence = perfect (100%)
    null_x = np.array([1, 10])
    null_y = np.array([50, 100])
    
    for level, ax in zip(['advanced', 'intermediate', 'beginner'], axes):
        # Extract data
        conf_data = extract_confidence(response_levels[level])
        score_data = score_levels[level]
        accuracy_data = (score_data / num_total) * 100  # Convert to percentage
        color = COLORS[level]
        
        # Scatter plot with transparency
        ax.scatter(conf_data, accuracy_data, color=color, alpha=0.1, s=50, 
                  edgecolors='none', label='Participants')
        
        # Calculate line of best fit
        coeffs = np.polyfit(conf_data, accuracy_data, 1)
        slope, intercept = coeffs
        
        # Generate line
        x_line = np.linspace(1, 10, 100)
        y_line = slope * x_line + intercept
        
        # Calculate R-squared
        y_pred = slope * conf_data + intercept
        ss_res = np.sum((accuracy_data - y_pred) ** 2)
        ss_tot = np.sum((accuracy_data - np.mean(accuracy_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard error of the fit
        n = len(conf_data)
        residuals = accuracy_data - y_pred
        s_residual = np.sqrt(np.sum(residuals**2) / (n - 2))
        x_mean = np.mean(conf_data)
        se_line = s_residual * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((conf_data - x_mean)**2))
        ci_line = 1.96 * se_line
        
        # Plot line of best fit with CI
        ax.plot(x_line, y_line, color='darkred', linewidth=PLOT_STYLE['mean_line_width'],
               label=f'Best fit: y={slope:.2f}x+{intercept:.1f} (R²={r_squared:.3f})')
        ax.fill_between(x_line, y_line - ci_line, y_line + ci_line, 
                        color='darkred', alpha=0.2, label='95% CI')
        
        # Plot null hypothesis line
        ax.plot(null_x, null_y, color=COLORS['null_hypothesis'], 
               linestyle=PLOT_STYLE['null_line_style'], 
               linewidth=PLOT_STYLE['null_line_width'],
               label='Null hypothesis')
        
        # Formatting
        ax.set_xlabel('Self-Confidence Rating (1 = least, 10 = most)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{level.capitalize()} (n={len(conf_data)})', 
                    fontsize=13, fontweight='bold')
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=PLOT_STYLE['grid_alpha'], linestyle='--')
        ax.legend(loc='lower right', fontsize=10)
    
    plt.suptitle('Self-Confidence vs Actual Accuracy', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig
    
# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    csv_file = "raw_data.csv"
    responses = import_and_clean_csv(csv_file)

    # Clean experience level labels
    responses[:,0][responses[:,0]=="I've folded Satoshi Kamiya's Ancient Dragon (or similar difficulty)"] = "advanced"
    responses[:,0][responses[:,0]=="Some experience with simple models"] = "intermediate"
    responses[:,0][responses[:,0]=="No experience"] = "beginner"
    
    num_real = len(REAL_MODELS)
    num_ai = len(AI_MODELS)
    num_total = num_real + num_ai

    # Calculate scores
    scores = responses.copy()
    scores[:,1:num_real+1][scores[:,1:num_real+1]=="Real"] = "1"
    scores[:,1:num_real+1][scores[:,1:num_real+1]=="AI"] = "0"
    scores[scores == "Not sure"] = str(NOT_SURE)
    scores[:,num_real+1:num_real+num_ai+1][scores[:,num_real+1:num_real+num_ai+1]=="AI"] = "1"
    scores[:,num_real+1:num_real+num_ai+1][scores[:,num_real+1:num_real+num_ai+1]=="Real"] = "0"
    
    # Save cleaned data
    column_headers = ["experience"] + REAL_MODELS + AI_MODELS + ["confidence"]
    np.savetxt("cleaned_data.csv", np.vstack([np.array(column_headers), scores]), 
               delimiter=",", fmt="%s")

    scores = np.sum(scores[:, 1:num_real+num_ai+1].astype(float), axis=1)

    # Split data by experience level
    response_levels, score_levels = split_by_experience(responses, scores)

    # Extract comments
    extract_comments("raw_data.csv")

    # Generate all plots
    fig1 = plot_score_histograms(responses, scores, score_levels)
    fig1.savefig('plots/accuracy_histograms.png', dpi=300, bbox_inches='tight')

    fig2 = create_stacked_bar_chart(response_levels['advanced'], 
                                     "Advanced")
    fig3 = create_stacked_bar_chart(response_levels['intermediate'], 
                                     "Intermediate")
    fig4 = create_stacked_bar_chart(response_levels['beginner'], 
                                     "Beginner")
    
    fig2.savefig('plots/classification_advanced.png', dpi=300, bbox_inches='tight')
    fig3.savefig('plots/classification_intermediate.png', dpi=300, bbox_inches='tight')
    fig4.savefig('plots/classification_beginner.png', dpi=300, bbox_inches='tight')
    
    fig5 = plot_skew_histograms(response_levels)
    fig5.savefig('plots/skew_histograms.png', dpi=300, bbox_inches='tight')
    
    fig6 = plot_confidence_histograms(response_levels)
    fig6.savefig('plots/confidence_histograms.png', dpi=300, bbox_inches='tight')
    fig7 = plot_stacked_confidence_histogram(response_levels, COLORS)
    fig7.savefig('plots/confidence_histogram_stacked.png', dpi=300, bbox_inches='tight')

    fig8 = plot_confidence_vs_accuracy(response_levels, score_levels)
    fig8.savefig('plots/confidence_vs_accuracy.png', dpi=300, bbox_inches='tight')

    print("All plots generated successfully!")