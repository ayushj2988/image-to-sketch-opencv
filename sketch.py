import cv2
import numpy as np

def image_to_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    sketch = cv2.divide(gray, blur, scale=256)
    return sketch
