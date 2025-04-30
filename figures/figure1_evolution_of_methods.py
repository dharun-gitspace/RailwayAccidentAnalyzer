import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure with specified size
plt.figure(figsize=(14, 7))

# Setup the axis
ax = plt.gca()
ax.set_ylim([0, 10])
ax.set_xlim([1950, 2030])

# Remove axis spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks([])  # No y-axis ticks

# Set x-axis ticks at decade intervals
decades = np.arange(1950, 2031, 10)
ax.set_xticks(decades)
ax.set_xticklabels([str(d) for d in decades], fontsize=12)

# Create timeline
plt.axhline(y=5, xmin=0, xmax=1, color='black', alpha=0.3, linestyle='-', linewidth=2)

# Define eras with methods
eras = [
    {"period": "1950s-1960s", "methods": ["Statistical Analysis", "Frequency Counting"], "y": 5, "x": 1955, "color": "#FFD700"},
    {"period": "1960s-1970s", "methods": ["Regression Analysis", "Probability Models"], "y": 5, "x": 1965, "color": "#FF9966"},
    {"period": "1970s-1980s", "methods": ["Risk Assessment", "Fault Tree Analysis"], "y": 5, "x": 1975, "color": "#FF6347"},
    {"period": "1980s-1990s", "methods": ["GIS Mapping", "System Safety Analysis"], "y": 5, "x": 1985, "color": "#CD5C5C"},
    {"period": "1990s-2000s", "methods": ["Bayesian Networks", "Neural Networks"], "y": 5, "x": 1995, "color": "#8A2BE2"},
    {"period": "2000s-2010s", "methods": ["Data Mining", "Support Vector Machines"], "y": 5, "x": 2005, "color": "#4682B4"},
    {"period": "2010s-2020s", "methods": ["Deep Learning", "Big Data Analytics"], "y": 5, "x": 2015, "color": "#1E90FF"},
    {"period": "2020s-Present", "methods": ["Integrated ML Approaches", "Explainable AI"], "y": 5, "x": 2025, "color": "#00BFFF"}
]

# Plot each era as a point on the timeline with a rectangle around methods
for era in eras:
    # Plot point on timeline
    plt.plot(era["x"], era["y"], 'o', markersize=15, color=era["color"])
    
    # Add period label above
    plt.text(era["x"], era["y"]+1.5, era["period"], ha='center', fontsize=12, fontweight='bold')
    
    # Add methods below in a box
    methods_text = "\n".join(era["methods"])
    ypos = era["y"] - 2
    
    # Calculate box dimensions
    method_lines = len(era["methods"])
    box_height = 0.8 * method_lines + 0.4
    box_width = max([len(m) for m in era["methods"]]) * 0.5 + 1
    
    # Create rectangle
    rect = patches.Rectangle((era["x"]-box_width/2, ypos-box_height/2+0.2), 
                           box_width, box_height, 
                           linewidth=1, edgecolor=era["color"], facecolor='white', alpha=0.7)
    ax.add_patch(rect)
    
    # Add methods text
    plt.text(era["x"], ypos, methods_text, ha='center', va='center', fontsize=10)

# Show progression arrows between eras
for i in range(len(eras)-1):
    plt.annotate('',
                xy=(eras[i+1]["x"]-2, 5), xycoords='data',
                xytext=(eras[i]["x"]+2, 5), textcoords='data',
                arrowprops=dict(arrowstyle="->", lw=1.5, color='gray'))

# Add title
plt.title('Figure 1: Evolution of Railway Safety Analysis Methods (1950-Present)', fontsize=16, pad=20)

# Save the figure
plt.tight_layout()
plt.savefig('figure1_evolution_of_methods.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 created successfully!")