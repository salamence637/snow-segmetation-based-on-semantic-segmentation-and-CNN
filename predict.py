import cv2
import numpy as np
from PIL import Image
from unet import Unet

# to examine if a single image is correct
unet = Unet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = unet.detect_image(image)
        r_image.show()
