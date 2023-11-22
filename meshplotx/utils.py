import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional


# Helper functions
def get_colors(
    inp: np.ndarray,
    colormap: str = "viridis",
    v_range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """
    inp: (...)
    returns: (..., 3)
    """
    cmap = plt.cm.get_cmap(colormap)
    if v_range is None:
        vmin = np.min(inp)
        vmax = np.max(inp)
    else:
        vmin, vmax = v_range

    norm = Normalize(vmin, vmax)
    return cmap(norm(inp))[..., :3]


def gen_checkers(n_checkers_x: int, n_checkers_y: int, width: int = 256, height: int = 256) -> np.ndarray:
    """
    returns: (width, height, 3)
    """
    # tex dims need to be power of two.
    array = np.ones((width, height, 3), dtype="float32")

    # width in texels of each checker
    checker_w = width / n_checkers_x
    checker_h = height / n_checkers_y

    for y in range(height):
        for x in range(width):
            color_key = int(x / checker_w) + int(y / checker_h)
            if color_key % 2 == 0:
                array[x, y, :] = [1.0, 0.874, 0.0]
            else:
                array[x, y, :] = [0.0, 0.0, 0.0]
    return array


def gen_circle(width: int = 256, height: int = 256) -> np.ndarray:
    """
    returns: (width, height, 4)
    """
    xx, yy = np.mgrid[:width, :height]
    circle = (xx - width / 2 + 0.5) ** 2 + (yy - height / 2 + 0.5) ** 2
    array = np.ones((width, height, 4), dtype="float32")
    array[:, :, 0] = circle <= width
    array[:, :, 1] = circle <= width
    array[:, :, 2] = circle <= width
    array[:, :, 3] = circle <= width
    return array


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
