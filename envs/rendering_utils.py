# pylint:disable-all

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os

IM_SCALE = 1.

def load_asset(asset_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, 'assets')
    asset_path = os.path.join(asset_dir_path, asset_name)
    return plt.imread(asset_path)

def fig2data(fig):
    fig.set_dpi(150)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]
    return data

def initialize_figure(height, width, fig_scale=1., background_grid=True):
    fig = plt.figure(figsize=((width + 2) * fig_scale, (height + 2) * fig_scale))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    # Draw a grid in the background
    if background_grid:
        for r in range(height):
            for c in range(width):
                edge_color = '#888888'
                face_color = 'white'
                drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                             numVertices=4,
                                             radius=0.5 * np.sqrt(2),
                                             orientation=np.pi / 4,
                                             ec=edge_color,
                                             fc=face_color)
                ax.add_patch(drawing)

    return fig, ax

def render_from_fig(fig):
    im = fig2data(fig)
    plt.close(fig)

    im = Image.fromarray(im)
    new_width, new_height = (int(im.size[0] * IM_SCALE), int(im.size[1] * IM_SCALE))
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    im = np.array(im)

    return im

def render_from_layout(layout, get_token_images, background_grid=True,
                       line_every=None):
    height, width = layout.shape[:2]

    fig, ax = initialize_figure(height, width, background_grid=background_grid)

    for r in range(height):
        for c in range(width):
            token_images = get_token_images(layout[r, c])
            for im in token_images:
                draw_token(im, r, c, ax, height, width)
    if line_every is not None:
        for r in range(height):
            if r > 0 and r % line_every == 0:
                ax.axhline(r)
        for c in range(width):
            if c > 0 and c % line_every == 0:
                ax.axvline(c)

    return render_from_fig(fig)

def draw_token(token_image, r, c, ax, height, width, token_scale=1.0, fig_scale=1.0):
    oi = OffsetImage(token_image, zoom = fig_scale * (token_scale / max(height, width)**0.5))
    box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)
    ax.add_artist(box)
    return box
