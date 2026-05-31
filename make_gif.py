import json
import re
import io  # Added missing import
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

html_filename = "final_complete_urban_scene.html"
output_gif = "urban_scene.gif"

print("Reading data coordinates directly from HTML file...")
with open(html_filename, "r", encoding="utf-8") as f:
    html_content = f.read()

print("Extracting plot configuration matrix...")
data_match = re.search(
    r'Plotly\.newPlot\(\s*[\'"][^\'"]+[\'"]\s*,\s*(\[\s*\{.*?\}\s*\])\s*,',
    html_content,
    re.DOTALL,
)
if not data_match:
    data_match = re.search(
        r"window\.PLOTLYENV\s*=.*?data\s*:\s*(\[\s*\{.*?\}\s*\])",
        html_content,
        re.DOTALL,
    )

data_json = json.loads(data_match.group(1))

trace = data_json[0]
X = np.array(trace["x"])
Y = np.array(trace["y"])
Z = np.array(trace["z"])

print(f"Extracted {len(X)} points. Parsing color properties...")

# Direct extraction from the line or marker sub-blocks
point_colors = None
if "marker" in trace:
    marker = trace["marker"]
    if "color" in marker:
        point_colors = np.array(marker["color"])
    elif "colorscale" in marker:
        point_colors = Z

if point_colors is None:
    point_colors = Z

print("Initializing maximized canvas layout...")
# Balanced square figure context
fig = plt.figure(figsize=(8, 8), dpi=150)
ax = fig.add_subplot(111, projection="3d")

# Zero out margins inside the Matplotlib structure
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_axis_off()

fig.patch.set_facecolor("#0b0f19")
ax.set_facecolor("#0b0f19")

print("Plotting dense point cloud scene...")
scatter = ax.scatter(
    X, Y, Z, c=point_colors, cmap="jet", s=0.6, marker="o", edgecolors="none", alpha=1.0
)

# Calculate spatial bounds tightly
max_range = (
    np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
)
mid_x = (X.max() + X.min()) / 2.0
mid_y = (Y.max() + Y.min()) / 2.0
mid_z = (Z.max() + Z.min()) / 2.0

# Using 0.65 pushes the axes inward to act as a tight zoom lens on the scene
zoom_factor = 0.65
ax.set_xlim(mid_x - max_range * zoom_factor, mid_x + max_range * zoom_factor)
ax.set_ylim(mid_y - max_range * zoom_factor, mid_y + max_range * zoom_factor)
ax.set_zlim(mid_z - max_range * zoom_factor, mid_z + max_range * zoom_factor)

frames = []
num_frames = 45

print("Rendering zoomed 360-degree rotation angles...")
plt.ioff()

for i in range(num_frames):
    angle = (i / num_frames) * 360
    ax.view_init(elev=24, azim=angle)

    # Save directly to memory buffer while explicitly clipping edge padding space
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0,
    )
    buf.seek(0)

    frames.append(imageio.imread(buf))

plt.close(fig)

print(f"Compiling frames into optimized {output_gif}...")
imageio.mimsave(output_gif, frames, duration=0.06, loop=0)
print("Complete. Your maximized, full-frame GIF is ready.")
