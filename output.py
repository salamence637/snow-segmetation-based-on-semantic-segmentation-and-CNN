import csv
import cv2
import numpy as np
from PIL import Image
from unet import Unet
unet = Unet()
def snowpercentage(image):
    img = np.array(image)
    rows,cols,channels = img.shape
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    rows, cols, channels = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([250, 250, 250])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower, upper)

    s = mask[0:1024, 0:768]  # y,x
    cv2.rectangle(img, (0, 0), (768, 1024), (0, 0, 255), 3)
    x, y = s.shape

    bk = 0
    wt = 0
    for i in range(x):
        for j in range(y):
            if s[i, j] == 0:
                bk += 1
            else:
                wt += 1
    rate1 = wt / (x * y)
    return rate1

    # cv2.waitKey(0)

text_path = 'list.txt'
img_path = []
SnowCoverage = []
with open(text_path,'r') as file:
    path = file.readlines()
    n_path = len(path)
    # print(n_path)
    for i,path in enumerate(path):
        path = path.strip('\n')
        try:
            image = Image.open(path)
            img_path.append(path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = unet.detect_image(image)
            SnowCoverage.append(snowpercentage(r_image))
            print(img_path[i],SnowCoverage[i])
            # r_image.show()
            # print(np.asarray(r_image))

csv_path = 'result/output.csv'
with open(csv_path,'w',newline='') as f:
    fieldnames = ['Path','SnowCoverage']
    f_csv = csv.DictWriter(f,fieldnames=fieldnames)
    f_csv.writeheader()
    for i in range(0,len(img_path)):
        f_csv.writerow({
            'Path':img_path[i],
            'SnowCoverage':SnowCoverage[i]
        })
