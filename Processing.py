import tensorflow as tf
from PIL import Image
import numpy as np

image = Image.open('train_2227431.tif')
img_array = np.array(image)

def listup_files(path):
    yield [os.path.abspath(p) for p in glob.glob(path)]

