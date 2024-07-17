import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
print(torch.__path__)
print(torch.__version__)
print(torchvision.__version__)
#response = requests.get(url)
img_path = '/home/deep/ur3e_gesture_estimate/result/img/ur3e-image2-00001.png'
try:
    image = Image.open(img_path)
    image.show()  # 或者使用 image.load() 来确保图像可以被加载
    print("Image loaded successfully!")
except Exception as e:
    print(f"Failed to load image: {e}")