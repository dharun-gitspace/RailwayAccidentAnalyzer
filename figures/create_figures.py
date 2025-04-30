import matplotlib.pyplot as plt
import numpy as np

# Create directory for finished figures
import os
if not os.path.exists('finished_figures'):
    os.makedirs('finished_figures')

#---------- FIGURE 1: TIMELINE OF METHODS ----------#
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

# Plot each era as a point on the timeline
for era in eras:
    # Plot point on timeline
    plt.plot(era["x"], era["y"], 'o', markersize=15, color=era["color"])
    
    # Add period label above
    plt.text(era["x"], era["y"]+1.5, era["period"], ha='center', fontsize=12, fontweight='bold')
    
    # Add methods below in a text box
    methods_text = "\n".join(era["methods"])
    plt.text(era["x"], era["y"]-2, methods_text, 
             ha='center', va='center', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor=era["color"]))

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
plt.savefig('finished_figures/figure1_evolution_of_methods.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 created successfully!")

#---------- FIGURE 2: GEOGRAPHICAL DISTRIBUTION ----------#
plt.figure(figsize=(14, 8))

# Create a simple world map representation
# This is a simplified approach using rectangles for continents
continents = {
    "North America": {"rect": [-150, -50, 60, 40], "color": "#FFD699"},
    "South America": {"rect": [-80, -20, 35, 60], "color": "#FFD699"},
    "Europe": {"rect": [-10, 40, 40, 30], "color": "#FFD699"},
    "Africa": {"rect": [-20, 40, 60, 70], "color": "#FFD699"},
    "Asia": {"rect": [40, 140, 10, 60], "color": "#FFD699"},
    "Australia": {"rect": [110, 155, -40, -10], "color": "#FFD699"}
}

# Study counts by region (number of railway safety studies)
study_counts = {
    "North America": 12,
    "South America": 3,
    "Europe": 15,
    "Africa": 4,
    "Asia": 25,  # India has the highest concentration
    "Australia": 5
}

# Set up a clean axis
ax = plt.gca()
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_aspect('equal')
ax.grid(linestyle='--', alpha=0.4)

# Remove regular axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Add simple coordinate grid
ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
ax.set_yticks([-60, -30, 0, 30, 60])
ax.set_xticklabels(['150°W', '100°W', '50°W', '0°', '50°E', '100°E', '150°E'])
ax.set_yticklabels(['60°S', '30°S', 'Equator', '30°N', '60°N'])

# Plot continents as colored rectangles
for name, data in continents.items():
    x1, x2, y1, y2 = data["rect"]
    width = x2 - x1
    height = y2 - y1
    # Calculate color intensity based on study count
    alpha = 0.3 + (study_counts[name] / max(study_counts.values())) * 0.7
    rect = plt.Rectangle((x1, y1), width, height, 
                         fc=data["color"], ec='black', alpha=alpha)
    ax.add_patch(rect)
    
    # Add region name and study count
    plt.text(x1 + width/2, y1 + height/2, 
             f"{name}\n{study_counts[name]} studies", 
             ha='center', va='center', fontweight='bold')

# Add a special marker for India (highest focus)
plt.plot(78, 22, 'o', markersize=15, color='blue')
plt.text(78, 30, "India\n(Focus of Current Study)", 
         ha='center', va='center', fontsize=12, 
         fontweight='bold', color='blue')

# Add connecting lines to show research flow
for region, data in continents.items():
    if region != "Asia":
        x1, x2, y1, y2 = data["rect"]
        center_x = x1 + (x2 - x1)/2
        center_y = y1 + (y2 - y1)/2
        plt.arrow(center_x, center_y, 
                  (78 - center_x)*0.8, (22 - center_y)*0.8,
                  head_width=5, head_length=5, fc='blue', ec='blue', alpha=0.5)

# Add legend for study density
cmap = plt.cm.Blues
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), 
             ax=ax,
             label='Number of Railway Safety Studies',
             orientation='horizontal',
             pad=0.05,
             shrink=0.6)

# Add title and note
plt.title('Figure 2: Geographical Distribution of Previous Railway Safety Studies', fontsize=16, pad=20)
plt.figtext(0.5, 0.01, 
            "Note: Darkness of shading indicates concentration of studies. India shows the highest focus in recent literature.",
            ha='center', fontsize=10, style='italic')

# Save the figure
plt.tight_layout()
plt.savefig('finished_figures/figure2_geographical_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2 created successfully!")
print("All figures created and saved to 'finished_figures' directory")