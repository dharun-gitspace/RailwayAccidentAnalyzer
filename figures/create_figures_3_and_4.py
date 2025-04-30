import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Create directory for figures
import os
if not os.path.exists('finished_figures'):
    os.makedirs('finished_figures')

#---------- FIGURE 3: APPLICATION ARCHITECTURE DIAGRAM ----------#
plt.figure(figsize=(12, 8))

# Set up a clean axis
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')  # Hide axis

# Define colors
colors = {
    'module': '#3498db',       # Blue
    'submodule': '#5DADE2',    # Light blue
    'data': '#f39c12',         # Orange
    'flow': '#2c3e50',         # Dark blue
    'user': '#27ae60',         # Green
    'border': '#34495e'        # Dark gray
}

# Draw the main modules
modules = [
    {'name': 'Data Processing Module', 'x': 2, 'y': 6, 'width': 6, 'height': 1.2},
    {'name': 'Analysis Module', 'x': 2, 'y': 4, 'width': 6, 'height': 1.2},
    {'name': 'Prediction Module', 'x': 1, 'y': 2, 'width': 3.5, 'height': 1.2},
    {'name': 'Visualization Module', 'x': 5.5, 'y': 2, 'width': 3.5, 'height': 1.2},
    {'name': 'User Interface (Streamlit)', 'x': 2, 'y': 0.5, 'width': 6, 'height': 0.8}
]

# Draw the modules
for module in modules:
    rect = patches.Rectangle(
        (module['x'], module['y']), 
        module['width'], module['height'],
        linewidth=1.5, 
        edgecolor=colors['border'], 
        facecolor=colors['module'],
        alpha=0.7
    )
    ax.add_patch(rect)
    
    # Add module name
    plt.text(
        module['x'] + module['width']/2, 
        module['y'] + module['height']/2,
        module['name'],
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold'
    )

# Add submodules inside Analysis Module
submodules = [
    {'name': 'Geospatial', 'x': 2.2, 'y': 4.25, 'width': 1.3, 'height': 0.5},
    {'name': 'Temporal', 'x': 3.8, 'y': 4.25, 'width': 1.3, 'height': 0.5},
    {'name': 'Anomaly', 'x': 5.4, 'y': 4.25, 'width': 1.3, 'height': 0.5},
    {'name': 'Severity', 'x': 7, 'y': 4.25, 'width': 0.8, 'height': 0.5}
]

for submodule in submodules:
    rect = patches.Rectangle(
        (submodule['x'], submodule['y']), 
        submodule['width'], submodule['height'],
        linewidth=1, 
        edgecolor=colors['border'], 
        facecolor=colors['submodule'],
        alpha=0.9
    )
    ax.add_patch(rect)
    
    # Add submodule name
    plt.text(
        submodule['x'] + submodule['width']/2, 
        submodule['y'] + submodule['height']/2,
        submodule['name'],
        ha='center',
        va='center',
        fontsize=10
    )

# Add data sources
data = {'name': 'Railway Accident Dataset', 'x': 2, 'y': 7.5, 'width': 6, 'height': 0.4}
rect = patches.Rectangle(
    (data['x'], data['y']), 
    data['width'], data['height'],
    linewidth=1, 
    edgecolor=colors['border'], 
    facecolor=colors['data'],
    alpha=0.7
)
ax.add_patch(rect)

# Add data source name
plt.text(
    data['x'] + data['width']/2, 
    data['y'] + data['height']/2,
    data['name'],
    ha='center',
    va='center',
    fontsize=11,
    fontweight='bold'
)

# Draw connecting arrows
arrows = [
    # Data to Processing
    {'start': (5, 7.5), 'end': (5, 7.2)},
    # Processing to Analysis
    {'start': (5, 6), 'end': (5, 5.2)},
    # Analysis to Prediction
    {'start': (3, 4), 'end': (3, 3.2)},
    # Analysis to Visualization
    {'start': (7, 4), 'end': (7, 3.2)},
    # Prediction to Visualization
    {'start': (4.5, 2.6), 'end': (5.5, 2.6)},
    # Visualization to UI
    {'start': (7, 2), 'end': (7, 1.3)},
    # Prediction to UI
    {'start': (3, 2), 'end': (3, 1.3)}
]

for arrow in arrows:
    plt.annotate(
        '',
        xy=arrow['end'], 
        xytext=arrow['start'],
        arrowprops=dict(
            facecolor=colors['flow'],
            shrink=0.05,
            width=1.5,
            headwidth=8,
            alpha=0.7
        )
    )

# Add user at the bottom
user = {'name': 'User', 'x': 4.5, 'y': 0.1, 'radius': 0.2}
circle = plt.Circle(
    (user['x'], user['y']), 
    user['radius'],
    color=colors['user'],
    alpha=0.7
)
ax.add_patch(circle)

# Add user label
plt.text(
    user['x'] + 0.3, 
    user['y'],
    user['name'],
    ha='left',
    va='center',
    fontsize=10
)

# Add bidirectional arrow between user and UI
plt.annotate(
    '',
    xy=(5, 0.5), 
    xytext=(5, 0.2),
    arrowprops=dict(
        facecolor=colors['flow'],
        shrink=0.05,
        width=1.5,
        headwidth=8,
        alpha=0.7
    )
)
plt.annotate(
    '',
    xy=(5, 0.2), 
    xytext=(5, 0.5),
    arrowprops=dict(
        facecolor=colors['flow'],
        shrink=0.05,
        width=1.5,
        headwidth=8,
        alpha=0.7
    )
)

# Add legend
legend_elements = [
    patches.Patch(facecolor=colors['module'], edgecolor=colors['border'], alpha=0.7, label='Main Modules'),
    patches.Patch(facecolor=colors['submodule'], edgecolor=colors['border'], alpha=0.7, label='Analysis Components'),
    patches.Patch(facecolor=colors['data'], edgecolor=colors['border'], alpha=0.7, label='Data Source'),
    patches.Patch(facecolor=colors['flow'], alpha=0.7, label='Data Flow'),
    patches.Patch(facecolor=colors['user'], alpha=0.7, label='User Interaction')
]

plt.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=10
)

# Add title
plt.title('Figure 3: Railway Accidents Analysis Application Architecture', fontsize=16, pad=20)

# Save figure
plt.tight_layout()
plt.savefig('finished_figures/figure3_application_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 3 created successfully!")

#---------- FIGURE 4: USER INTERFACE LAYOUT ----------#
plt.figure(figsize=(14, 8))

# Set up a clean axis
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')  # Hide axis

# Define colors
colors = {
    'sidebar': '#f1c40f',      # Yellow
    'header': '#3498db',       # Blue
    'main': '#ecf0f1',         # Light gray
    'section': '#e74c3c',      # Red
    'border': '#34495e',       # Dark gray
    'text': '#2c3e50'          # Dark blue
}

# Draw main app container
main_app = {'x': 0.5, 'y': 0.5, 'width': 9, 'height': 7}
rect = patches.Rectangle(
    (main_app['x'], main_app['y']), 
    main_app['width'], main_app['height'],
    linewidth=2, 
    edgecolor=colors['border'], 
    facecolor=colors['main'],
    alpha=0.3
)
ax.add_patch(rect)

# Draw header
header = {'x': 0.5, 'y': 6.5, 'width': 9, 'height': 1}
rect = patches.Rectangle(
    (header['x'], header['y']), 
    header['width'], header['height'],
    linewidth=1, 
    edgecolor=colors['border'], 
    facecolor=colors['header'],
    alpha=0.7
)
ax.add_patch(rect)

# Add header text
plt.text(
    header['x'] + header['width']/2, 
    header['y'] + header['height']/2,
    'Indian Railway Accidents Analysis & Prediction',
    ha='center',
    va='center',
    fontsize=16,
    fontweight='bold',
    color='white'
)

# Draw sidebar
sidebar = {'x': 0.5, 'y': 0.5, 'width': 2.2, 'height': 6}
rect = patches.Rectangle(
    (sidebar['x'], sidebar['y']), 
    sidebar['width'], sidebar['height'],
    linewidth=1, 
    edgecolor=colors['border'], 
    facecolor=colors['sidebar'],
    alpha=0.5
)
ax.add_patch(rect)

# Add sidebar title
plt.text(
    sidebar['x'] + sidebar['width']/2, 
    sidebar['y'] + 5.7,
    'Navigation',
    ha='center',
    va='center',
    fontsize=14,
    fontweight='bold',
    color=colors['text']
)

# Add sidebar options
sidebar_options = [
    'Data Overview',
    'Severity Prediction',
    'Geospatial Analysis',
    'Temporal Trends',
    'Anomaly Detection'
]

for i, option in enumerate(sidebar_options):
    y_pos = sidebar['y'] + 5 - i*0.8
    
    # Add option button
    rect = patches.Rectangle(
        (sidebar['x'] + 0.2, y_pos - 0.3), 
        1.8, 0.6,
        linewidth=1, 
        edgecolor=colors['border'], 
        facecolor='white' if i != 2 else colors['section'],
        alpha=0.7,
        zorder=2
    )
    ax.add_patch(rect)
    
    # Add option text
    plt.text(
        sidebar['x'] + 0.2 + 0.9, 
        y_pos,
        option,
        ha='center',
        va='center',
        fontsize=10,
        color='black' if i != 2 else 'white',
        fontweight='normal' if i != 2 else 'bold',
        zorder=3
    )

# Draw main content area
content_area = {'x': 2.9, 'y': 0.7, 'width': 6.4, 'height': 5.6}
rect = patches.Rectangle(
    (content_area['x'], content_area['y']), 
    content_area['width'], content_area['height'],
    linewidth=1, 
    edgecolor=colors['border'], 
    facecolor='white',
    alpha=0.7
)
ax.add_patch(rect)

# Add main content section title
plt.text(
    content_area['x'] + 0.3, 
    content_area['y'] + 5.3,
    'Geospatial Analysis',
    ha='left',
    va='center',
    fontsize=14,
    fontweight='bold',
    color=colors['section']
)

# Add map visualization placeholder
map_area = {'x': 3.1, 'y': 2, 'width': 6, 'height': 3.5}
rect = patches.Rectangle(
    (map_area['x'], map_area['y']), 
    map_area['width'], map_area['height'],
    linewidth=1, 
    edgecolor=colors['border'], 
    facecolor='#d6e4ff',
    alpha=0.7
)
ax.add_patch(rect)

# Add map mockup elements - grid lines
for i in range(5):
    plt.plot(
        [map_area['x'], map_area['x'] + map_area['width']], 
        [map_area['y'] + i * map_area['height']/4, map_area['y'] + i * map_area['height']/4],
        color='gray', 
        alpha=0.3,
        linestyle=':'
    )
    
    plt.plot(
        [map_area['x'] + i * map_area['width']/4, map_area['x'] + i * map_area['width']/4], 
        [map_area['y'], map_area['y'] + map_area['height']],
        color='gray', 
        alpha=0.3,
        linestyle=':'
    )

# Add map points
np.random.seed(42)
point_colors = ['red', 'orange', 'orangered', 'darkred', 'red']
for i in range(20):
    x = np.random.uniform(map_area['x'] + 0.5, map_area['x'] + map_area['width'] - 0.5)
    y = np.random.uniform(map_area['y'] + 0.5, map_area['y'] + map_area['height'] - 0.5)
    size = np.random.uniform(30, 100)
    color = np.random.choice(point_colors)
    
    plt.scatter(x, y, s=size, color=color, alpha=0.7, edgecolors='black')

# Add map title
plt.text(
    map_area['x'] + map_area['width']/2, 
    map_area['y'] + map_area['height'] + 0.2,
    'Accident Hotspots Across India',
    ha='center',
    va='center',
    fontsize=12,
    fontweight='bold'
)

# Add controls section
controls_area = {'x': 3.1, 'y': 0.9, 'width': 6, 'height': 0.9}

# Add slider controls
plt.text(
    controls_area['x'], 
    controls_area['y'] + 0.7,
    'Cluster Radius (km):',
    ha='left',
    va='center',
    fontsize=10
)

rect = patches.Rectangle(
    (controls_area['x'] + 2, controls_area['y'] + 0.65), 
    3, 0.2,
    linewidth=1, 
    edgecolor=colors['border'], 
    facecolor='white',
    alpha=0.7
)
ax.add_patch(rect)

plt.text(
    controls_area['x'], 
    controls_area['y'] + 0.3,
    'Minimum Accidents:',
    ha='left',
    va='center',
    fontsize=10
)

rect = patches.Rectangle(
    (controls_area['x'] + 2, controls_area['y'] + 0.25), 
    3, 0.2,
    linewidth=1, 
    edgecolor=colors['border'], 
    facecolor='white',
    alpha=0.7
)
ax.add_patch(rect)

# Add apply button
rect = patches.Rectangle(
    (controls_area['x'] + 5.2, controls_area['y'] + 0.45), 
    0.7, 0.3,
    linewidth=1, 
    edgecolor=colors['border'], 
    facecolor=colors['section'],
    alpha=0.7
)
ax.add_patch(rect)

plt.text(
    controls_area['x'] + 5.55, 
    controls_area['y'] + 0.6,
    'Apply',
    ha='center',
    va='center',
    fontsize=10,
    color='white'
)

# Add legend for map
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, alpha=0.7, label='Cluster 1'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, alpha=0.7, label='Cluster 2'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10, alpha=0.7, label='Cluster 3'),
]

legend = plt.legend(
    handles=legend_elements,
    loc='lower right',
    bbox_to_anchor=(9.3, 2.2),
    fontsize=8,
    title='Accident Clusters'
)
legend.get_title().set_fontsize(9)

# Add annotated UI elements
plt.annotate(
    'Navigation Sidebar',
    xy=(1.6, 3.5), 
    xytext=(1.6, 2),
    arrowprops=dict(
        facecolor='black',
        shrink=0.05,
        width=1,
        headwidth=5,
        alpha=0.6
    ),
    ha='center',
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
)

plt.annotate(
    'Interactive Map',
    xy=(6, 3.5), 
    xytext=(8.5, 4),
    arrowprops=dict(
        facecolor='black',
        shrink=0.05,
        width=1,
        headwidth=5,
        alpha=0.6
    ),
    ha='center',
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="blue", alpha=0.3)
)

plt.annotate(
    'Control Panel',
    xy=(6, 1.1), 
    xytext=(8.5, 1.5),
    arrowprops=dict(
        facecolor='black',
        shrink=0.05,
        width=1,
        headwidth=5,
        alpha=0.6
    ),
    ha='center',
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.3)
)

# Add title
plt.title('Figure 4: User Interface Layout - Geospatial Analysis Section', fontsize=16, pad=20)

# Save figure
plt.tight_layout()
plt.savefig('finished_figures/figure4_user_interface_layout.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 created successfully!")
print("All figures created successfully!")