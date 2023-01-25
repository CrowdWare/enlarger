import cv2, os
from cv2 import dnn_superres
from glob import glob

image_files = glob(os.path.join('images', '*.*'))
sr = dnn_superres.DnnSuperResImpl_create()
path = "data/ESPCN_x4.pb"
sr.readModel(path)
sr.setModel("espcn", 4)

for file in image_files:
    base = os.path.basename(file).split('.', 1)[0]
    print("processing:", base)
    image = cv2.imread(file)
    result = sr.upsample(image)
    cv2.imwrite("upscaled/" + base + "_x4.jpg", result)