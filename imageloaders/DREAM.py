
import os
import time
import sys
import os
base_dir = os.path.abspath("/home/deep/CtRNet-robot-pose-estimation")
sys.path.append(base_dir)
from PIL import Image
from tqdm.notebook import tqdm
import cv2
import numpy as np
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import glob
import pickle
import json
from utils import find_ndds_data_in_dir, transform_DREAM_to_CPLSim_TCR
import transforms3d.quaternions as tq 
import numpy as np

def quaternion_and_translation_to_transform_matrix(quaternion, translation):
    # 使用 transforms3d 中的 quaternion_matrix 函数
    transform_matrix = tq.quat2mat(quaternion)
    # 创建一个4x4变换矩阵
    transform_matrix_4x4 = np.eye(4)
    transform_matrix_4x4[:3, :3] = transform_matrix
    # 设置平移部分
    transform_matrix_4x4[:3, 3] = translation
    return transform_matrix_4x4

def base_to_inertia():
    
    transform_matrix_4x4 = np.eye(4)
    transform_matrix_4x4[0, 0] = -1
    transform_matrix_4x4[1, 1] = -1
    # 设置平移部分
    
    return np.linalg.inv(transform_matrix_4x4)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_camera_parameters(data_folder):
    _, ndds_data_configs = find_ndds_data_in_dir(data_folder)
    with open(ndds_data_configs['camera'], "r") as json_file:
        data = json.load(json_file)

    fx = data['camera_settings'][0]['intrinsic_settings']['fx']
    fy = data['camera_settings'][0]['intrinsic_settings']['fy']
    cx = data['camera_settings'][0]['intrinsic_settings']['cx']
    cy = data['camera_settings'][0]['intrinsic_settings']['cy']

    return fx, fy, cx, cy


class ImageDataLoaderSynthetic(Dataset):

    def __init__(self, data_folder, scale=1, trans_to_tensor=None):
        self.trans_to_tensor = trans_to_tensor
        self.data_folder = data_folder

        self.ndds_dataset, self.ndds_data_configs = find_ndds_data_in_dir(self.data_folder)
        print(self.ndds_dataset)
        self.scale = scale



    def __len__(self):

        return len(self.ndds_dataset)

    def __getitem__(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        if "panda" in self.data_folder:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            TCR_ndds = np.array(data['objects'][0]['pose_transform'])
            base_to_cam = transform_DREAM_to_CPLSim_TCR(TCR_ndds)

            joint_angle = torch.tensor(joint_angle, dtype=torch.float)
            base_to_cam = torch.tensor(base_to_cam, dtype=torch.float)

        else:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            TCR_ndds = np.array(data['objects'][0]['pose_transform'])
            base_to_cam = transform_DREAM_to_CPLSim_TCR(TCR_ndds)

            joint_angle = torch.tensor(joint_angle, dtype=torch.float)
            base_to_cam = torch.tensor(base_to_cam, dtype=torch.float)


        return image, joint_angle, base_to_cam

class ImageDataLoaderSyntheticCopp(Dataset):

    def __init__(self, img_folder, csv_path, scale=1, trans_to_tensor=transforms.Compose([transforms.ToTensor()])):
        self.trans_to_tensor = trans_to_tensor
        self.img_folder = img_folder
        self.csv_path = csv_path
        #self.ndds_dataset, self.ndds_data_configs = find_ndds_data_in_dir(self.data_folder)

        self.scale = scale
        self.csvdata = []
        df = pd.read_csv(self.csv_path, header=None, names=['id', 'joint_rad', 'camera_ex_parameter'])
        for _, row in df.iterrows():
            img_number = row['id']
            features_1 = eval(row['joint_rad'])  
            features_2 = eval(row['camera_ex_parameter']) 
            self.csvdata.append((img_number, features_1, features_2))
        #print(len(self.csvdata))
       # print(img_number)
        # self.img_names = sorted(
        #     [img_name for img_name in os.listdir(self.img_folder) if img_name.endswith('.png')],
        #     key=lambda x: int(x.split('-')[-1].split('.')[0])
        # )
        self.img_names = [img_name for img_name in os.listdir(self.img_folder) if img_name.endswith('.png')]
       # print(self.img_names)

    def __len__(self):

        return len(self.img_names)

    def __getitem__(self, idx):

        img_name = self.img_names[idx]
        #print(img_name)
        # load image
        img_path = os.path.join(self.img_folder, img_name)
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        if self.trans_to_tensor:
            image = self.trans_to_tensor(image_pil)
        else: 
            image = image_pil
        img_number = int(img_name.split('-')[-1].split('.')[0])    
        
        #img_number = 1
        joint_angle, base_to_cam = None, None
        
        if self.csvdata[img_number-1][0] == img_number:
            joint_angle = torch.tensor(self.csvdata[img_number-1][1], dtype=torch.double)
            q = self.csvdata[img_number-1][2][3:7]
            
            matrix = quaternion_and_translation_to_transform_matrix(np.array([q[3], q[0], q[1], q[2]]), self.csvdata[img_number-1][2][:3])
            base_to_cam = torch.tensor(matrix, dtype=torch.double)
            base_to_cam[:3,3] = base_to_cam[:3,3] 
            #basetointertia = base_to_inertia()
            basetointertia = np.eye(4)
            intertia = torch.tensor(basetointertia, dtype=torch.double)
            # in coppeliasim we got T inertia to camera we need base_link to camera
        return image, joint_angle, torch.tensor(np.dot(base_to_cam, intertia), dtype=torch.double)


class ImageDataLoaderReal(Dataset):

    def __init__(self, data_folder, scale=1, trans_to_tensor=None):
        self.trans_to_tensor = trans_to_tensor
        self.data_folder = data_folder

        self.ndds_dataset, self.ndds_data_configs = find_ndds_data_in_dir(self.data_folder)
        print(self.ndds_dataset)
        print(self.ndds_data_configs)
        self.scale = scale



    def __len__(self):

        return len(self.ndds_dataset)

    def __getitem__(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        if "panda" in self.data_folder:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            joint_angle = torch.tensor(joint_angle, dtype=torch.float)


        else:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            joint_angle = torch.tensor(joint_angle, dtype=torch.float)



        return image, joint_angle

    def get_data_with_keypoints(self, idx):

        data_sample = self.ndds_dataset[idx]
        # load image
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = pil_loader(img_path)
        if self.scale != 1:
            width, height = image_pil.size
            new_size = (int(width*self.scale),int(height*self.scale))
            image_pil = image_pil.resize(new_size)
        image = self.trans_to_tensor(image_pil)

        label_json = data_sample["data_path"]
        with open(label_json, "r") as json_file:
            data = json.load(json_file)

        if "panda" in self.data_folder:
            joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                    data['sim_state']['joints'][1]['position'],
                                    data['sim_state']['joints'][2]['position'],
                                    data['sim_state']['joints'][3]['position'],
                                    data['sim_state']['joints'][4]['position'],
                                    data['sim_state']['joints'][5]['position'],
                                    data['sim_state']['joints'][6]['position']])


            joint_angle = torch.tensor(joint_angle, dtype=torch.float)


        else:
            joint_angle = None

        keypoints = data['objects'][0]['keypoints']

        return image, joint_angle, keypoints
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import matplotlib.pyplot as plt

def main():
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd 
    import matplotlib.pyplot as plt
    data_folder = '/home/deep/ur3e_gesture_estimate/result/img/'
    csv_path = '/home/deep/ur3e_gesture_estimate/result/data/record.csv'
    test_data_folder = '/home/deep/ur3e_gesture_estimate/result/img/'
    scale = 0.25
    trans_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # dataset = ImageDataLoaderSyntheticCopp(img_folder=data_folder, csv_path=csv_path, scale=scale, trans_to_tensor=trans_to_tensor)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    for phase in ['train', 'valid']:
        datasets[phase] = ImageDataLoaderSyntheticCopp(img_folder = data_folder if phase=='train' else test_data_folder, csv_path=csv_path if phase=='train' else csv_path, scale = scale, trans_to_tensor = trans_to_tensor)
        
        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=32,
            shuffle=True if phase == 'train' else False,
            num_workers=8)

        data_n_batches[phase] = len(dataloaders[phase])
        loader = dataloaders[phase]

        #bar = ProgressBar(maxval=data_n_batches[phase])
        # for i, data in tqdm(enumerate(loader), total=data_n_batches[phase]):
        #     print(data_n_batches)

    
    print(data_n_batches)
    
    image, joint, t = datasets['train'][0]
    print(joint)
    print(t)
    image = transforms.ToPILImage()(image)
        
    # 显示图片
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')
    plt.show()

def main2():
    trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_folder = '/home/deep/CtRNet-robot-pose-estimation/panda_synth_test_dr/panda_synth_test_dr'
    test_data_folder = '/home/deep/CtRNet-robot-pose-estimation/panda_synth_test_dr/panda_synth_test_dr'
    datasets = ImageDataLoaderSynthetic(data_folder = data_folder, scale = 0.5, trans_to_tensor = trans_to_tensor)
    print(datasets)
    # datasets = {}
    # dataloaders = {}
    # data_n_batches = {}
    # for phase in ['train','valid']:
    #     datasets[phase] = ImageDataLoaderSynthetic(data_folder = data_folder if phase=='train' else test_data_folder, scale = 0.5, trans_to_tensor = trans_to_tensor)
    #     print(datasets[phase])

    #     dataloaders[phase] = DataLoader(
    #         datasets[phase], batch_size=16,
    #         shuffle=True if phase == 'train' else False,
    #         num_workers=16)

    #     data_n_batches[phase] = len(dataloaders[phase])

if __name__ == "__main__":
    main2()
# datasets = {}
# dataloaders = {}
# data_n_batches = {}
# for phase in ['train', 'valid']:
#     datasets[phase] = ImageDataLoaderSyntheticCopp(img_folder = data_folder if phase=='train' else test_data_folder, csv_path=csv_path, scale = scale, trans_to_tensor = trans_to_tensor)
#     print(phase)

#     # dataloaders[phase] = DataLoader(
#     #     datasets[phase], batch_size=32,
#     #     shuffle=False if phase == 'train' else False,
#     #     num_workers=8)

#   #  data_n_batches[phase] = len(dataloaders[phase])
# print(dataloaders)
# for i, data, t in enumerate(dataloaders['train']):
#     joint_angle = data
#     print(joint_angle)
#     break

    
