import cv2 
import matplotlib.pyplot as plt
import numpy as np


def cv2_imshow(img: np.ndarray):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
