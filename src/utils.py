import numpy as np
import contextlib
import io
import sys

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def rgb_to_gray(rgb):
    # scalar product of colors with certain theoretical coefficients according to the YUV system
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).round(3).astype(int)


