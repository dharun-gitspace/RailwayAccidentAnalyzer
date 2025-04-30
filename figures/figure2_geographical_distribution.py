import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection

# Create figure
plt.figure(figsize=(14, 8))

# Basic world map data (simplified for illustration)
# Note: This is a simplified approach for illustration purposes
# A real application would use proper GIS data libraries like geopandas
ax = plt.axes()

# Remove axis ticks and spines
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Define regions of interest with coordinates (simplified approximations)
regions = {
    "North America": [(-170, 15), (-50, 15), (-50, 85), (-170, 85)],
    "South America": [(-90, -60), (-30, -60), (-30, 15), (-90, 15)],
    "Europe": [(-10, 35), (40, 35), (40, 75), (-10, 75)],
    "Africa": [(-20, -40), (55, -40), (55, 35), (-20, 35)],
    "Asia": [(40, 0), (150, 0), (150, 75), (40, 75)],
    "Australia": [(110, -45), (160, -45), (160, -10), (110, -10)]
}

# Study distribution by region (number of studies)
study_counts = {
    "North America": 12,
    "South America": 3,
    "Europe": 15,
    "Africa": 4,
    "Asia": 25,  # India has the highest concentration
    "Australia": 5
}

# Calculate study density
max_count = max(study_counts.values())
min_count = min(study_counts.values())

# Create colormap
cmap = plt.cm.YlOrRd
norm = plt.Normalize(min_count, max_count)

# Draw regions with color based on study density
patches = []
colors = []
for region, coords in regions.items():
    polygon = Polygon(coords)  # Remove the second argument
    patches.append(polygon)
    colors.append(study_counts[region])

p = PatchCollection(patches, alpha=0.7)
p.set_array(np.array(colors))
p.set_cmap(cmap)
ax.add_collection(p)

# Add region labels
for region, coords in regions.items():
    centroid_x = sum([x for x, y in coords]) / len(coords)
    centroid_y = sum([y for x, y in coords]) / len(coords)
    plt.text(centroid_x, centroid_y, f"{region}\n({study_counts[region]} studies)", 
             ha='center', va='center', fontsize=10, fontweight='bold')

# Add special marker for India (highest concentration in Asia)
plt.plot(80, 25, 'o', markersize=15, color='blue')
plt.text(90, 25, "India\n(Focus of current study)", 
         ha='left', va='center', fontsize=10, fontweight='bold', color='blue')

# Create legend for study density
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label('Number of Railway Safety Studies')

# Add arrows to show the flow of research focus
plt.annotate('', xy=(80, 25), xytext=(60, 50), 
             arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8))
plt.annotate('', xy=(80, 25), xytext=(20, 65), 
             arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8))
plt.annotate('', xy=(80, 25), xytext=(-40, 55), 
             arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8))

# Add title
plt.title('Figure 2: Geographical Distribution of Previous Railway Safety Studies', fontsize=16, pad=20)

# Add note
plt.text(0, -50, "Note: Circle size indicates study concentration. India has the highest focus in recent literature.",
         ha='center', fontsize=10, style='italic')

# Save the figure
plt.tight_layout()
plt.savefig('figure2_geographical_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2 created successfully!")