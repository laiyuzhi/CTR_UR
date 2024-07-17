import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    PerspectiveCameras,Textures
)

from os.path import exists


class RobotMeshRenderer():
    """
    Class that render robot mesh with differentiable renderer
    """
    def __init__(self, focal_length, principal_point, image_size, robot, mesh_files, device):
        
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.device = device
        self.robot = robot
        self.mesh_files = mesh_files
        self.preload_verts = []
        self.preload_faces = []
        

        # preload the mesh to save loading time
        for m_file in mesh_files:
            assert exists(m_file)
            preload_verts_i, preload_faces_idx_i, _ = load_obj(m_file)
            preload_faces_i = preload_faces_idx_i.verts_idx
            self.preload_verts.append(preload_verts_i)
            self.preload_faces.append(preload_faces_i)


        # set up differentiable renderer with given camera parameters
        self.cameras = PerspectiveCameras(focal_length = [focal_length],
                                     principal_point = [principal_point],
                                     device=device, 
                                     in_ndc=False, image_size = [image_size]) #  (height, width) !!!!!
        
        blend_params = BlendParams(sigma=1e-8, gamma=1e-8)
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
            max_faces_per_bin=100000,  # max_faces_per_bin=1000000,  
        )
        
        # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        
        
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000, 
        )
        # We can add a point light in front of the object. 
        lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=self.cameras, lights=lights)
        )
        
    def get_robot_mesh(self, joint_angle):
        
        R_list, t_list = self.robot.get_joint_RT(joint_angle)
        assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

        verts_list = []
        faces_list = []
        verts_rgb_list = []
        verts_count = 0
        for i in range(len(self.mesh_files)):
            verts_i = self.preload_verts[i]
            faces_i = self.preload_faces[i]

            R = torch.tensor(R_list[i],dtype=torch.float32)
            t = torch.tensor(t_list[i],dtype=torch.float32)
            
            verts_i = verts_i @ R.T + t
            
            #verts_i = (R @ verts_i.T).T + t
            faces_i = faces_i + verts_count

            verts_count+=verts_i.shape[0]

            verts_list.append(verts_i.to(self.device))
            faces_list.append(faces_i.to(self.device))

            # Initialize each vertex to be white in color.
            color = torch.rand(3)
            verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
            verts_rgb_list.append(verts_rgb_i.to(self.device))



        verts = torch.concat(verts_list, dim=0)
        faces = torch.concat(faces_list, dim=0)

        verts_rgb = torch.concat(verts_rgb_list,dim=0)[None]
        textures = Textures(verts_rgb=verts_rgb)

        # Create a Meshes object
        robot_mesh = Meshes(
            verts=[verts.to(self.device)],   
            faces=[faces.to(self.device)], 
            textures=textures
        )
        
        return robot_mesh


    def get_robot_verts_and_faces(self, joint_angle):
        
        R_list, t_list = self.robot.get_joint_RT(joint_angle)
        assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

        verts_list = []
        faces_list = []
        verts_rgb_list = []
        verts_count = 0
        for i in range(len(self.mesh_files)):
            verts_i = self.preload_verts[i]
            faces_i = self.preload_faces[i]

            R = torch.tensor(R_list[i],dtype=torch.float32)
            t = torch.tensor(t_list[i],dtype=torch.float32)
            verts_i = verts_i @ R.T + t
            #verts_i = (R @ verts_i.T).T + t
            faces_i = faces_i + verts_count

            verts_count+=verts_i.shape[0]

            verts_list.append(verts_i.to(self.device))
            faces_list.append(faces_i.to(self.device))

            # Initialize each vertex to be white in color.
            #color = torch.rand(3)
            #verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
            #verts_rgb_list.append(verts_rgb_i.to(self.device))

        verts = torch.concat(verts_list, dim=0)
        faces = torch.concat(faces_list, dim=0)

        
        return verts, faces
    
    def render_robot_mesh(self, joint_angle, renderer_type='phong'):
        robot_mesh = self.get_robot_mesh(joint_angle)
        if renderer_type == 'phong':
            renderer = self.phong_renderer
        elif renderer_type == 'silhouette':
            renderer = self.silhouette_renderer
        else:
            raise ValueError("renderer_type must be 'phong' or 'silhouette'")
        
        # Render the mesh
        images = renderer(meshes_world=robot_mesh)
        return images

if __name__ == "__main__":
    # 示例数据和参数
    focal_length = [460,460]
    principal_point = [180, 360]
    image_size = [360,720]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 你需要提供正确的URDF解析类和mesh文件路径
    class DummyRobot:
        def get_joint_RT(self, joint_angle):
            # 示例返回值
            R_list = [np.eye(3)] * 7
            t_list = [np.ones(3)] * 7
            return np.array(R_list), np.array(t_list)
    
    robot = DummyRobot()
    mesh_files = ["/home/deep/CtRNet-robot-pose-estimation/urdfs/ur_description/meshes/ur3e/base/base.obj",
              "/home/deep/CtRNet-robot-pose-estimation/urdfs/ur_description/meshes/ur3e/shoulder/shoulder.obj",
              "/home/deep/CtRNet-robot-pose-estimation/urdfs/ur_description/meshes/ur3e/upperarm/upperarm.obj",
              "/home/deep/CtRNet-robot-pose-estimation/urdfs/ur_description/meshes/ur3e/forearm/forearm.obj",
              "/home/deep/CtRNet-robot-pose-estimation/urdfs/ur_description/meshes/ur3e/wrist1/wrist1.obj",
              "/home/deep/CtRNet-robot-pose-estimation/urdfs/ur_description/meshes/ur3e/wrist2/wrist2.obj",
              "/home/deep/CtRNet-robot-pose-estimation/urdfs/ur_description/meshes/ur3e/wrist3/wrist3.obj",
             ]

    renderer = RobotMeshRenderer(focal_length, principal_point, image_size, robot, mesh_files, device)
    joint_angle = np.zeros(6)  # 示例关节角度

    # 渲染并显示结果
    images = renderer.get_robot_mesh(joint_angle)
    image = images[0, ..., :3].cpu().numpy()
    
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()