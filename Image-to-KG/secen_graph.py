"""
Generates a circular Scene Graph visualization for Medical Images.
"""

import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Rectangle
from pathlib import Path

def draw_scene_graph(img_path, output_path, nodes, attributes):
    """
    Draws the scene graph on a canvas with the central image.
    """
    if not Path(img_path).exists():
        print(f" Image not found: {img_path}")
        return

    # Canvas Setup
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-8, 8)

    # Draw Central Image
    try:
        img = mpimg.imread(str(img_path))
        # Adjust extent to fit your aesthetic preference
        ax.imshow(img, extent=[-3, 3, -2.5, 2.5], zorder=1, cmap='gray')
        ax.add_patch(Rectangle((-3, -2.5), 6, 5, fill=False, lw=2, ec="black", zorder=2))
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Node Layout (Circular)
    N = len(nodes)
    R = 7.0
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    for idx, (node, angle) in enumerate(zip(nodes, angles)):
        x, y = R * np.cos(angle), R * np.sin(angle)

        # Edge (Center to Node)
        ax.plot([0, x], [0, y], color="#555", lw=1, zorder=0)

        # Node Circle
        ax.add_patch(Circle((x, y), 1.2, facecolor="#4A8BD8", edgecolor="#2C6BC3", lw=2, zorder=3))
        
        # Node Text
        display_text = "\n".join(textwrap.wrap(node, 10))
        ax.text(x, y, display_text, ha="center", va="center", 
                color="white", fontweight="bold", fontsize=10, zorder=4)

        # Attributes (Rectangles below node)
        attrs = attributes.get(node, [])
        for i, attr in enumerate(attrs):
            attr_y = y - 1.8 - (i * 0.8)
            ax.text(x, attr_y, attr, ha="center", va="center", 
                    fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFD6DE", ec="#C4586E"), 
                    zorder=5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close(fig)
    print(f"Scene Graph saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image (result of bbox script)")
    parser.add_argument("--output_path", type=str, default="scene_graph_output.png", help="Path to save the result")
    
    args = parser.parse_args()

    # ==========================================
    # [DEMO DATA]
    # In a real pipeline, load this from your model's output JSON.
    # ==========================================
    demo_nodes = [
        "Right lung", "Left lung", "Superior vena cava", 
        "Right lower zone", "Left lower zone", "Catheter"
    ]
    demo_attributes = {
        "Right lung": ["opacity"],
        "Left lung": ["clear"],
        "Catheter": ["present", "central line"]
    }

    draw_scene_graph(args.img_path, args.output_path, demo_nodes, demo_attributes)