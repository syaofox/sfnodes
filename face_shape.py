import torch
#import torch.nn.functional as F
import torchvision.transforms.v2 as T
import numpy as np
import comfy.utils
import comfy.model_management as mm
import gc
from tqdm import tqdm
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
import folder_paths
import os
import logging
from custom_nodes.sfnodes.media_pipe.mp_utils import LMKExtractor

_CATEGORY = 'sfnodes/face_analysis'

# 添加绘图功能
def draw_pointsOnImg(image, landmarks, color=(255, 0, 0), radius=3):
    # cv2.circle打坐标点的坐标系，左上角是原点，先写x再写y
    image_cpy = image.copy()
    for n in range(landmarks.shape[0]):
        try:
            cv2.circle(image_cpy, (int(landmarks[n][0]), int(landmarks[n][1])), radius, color, -1)                        
        except:
            pass
    return image_cpy

def drawLineBetweenPoints(image, pointsA, pointsB, color=(255, 0, 0), thickness=1):
    image_cpy = image.copy()
    for n in range(pointsA.shape[0]):
        try:
            cv2.line(image_cpy, (int(pointsA[n][0]), int(pointsA[n][1])), (int(pointsB[n][0]), int(pointsB[n][1])), color, thickness)                        
        except:
            pass
    return image_cpy


class FaceShaperMatch:
    def __init__(self):
        self.lmk_extractor = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "source_image": ("IMAGE",),
            "target_image": ("IMAGE",),  # 添加目标图像参数，替代target_crop_info          
            "landmarkType": (["ALL","OUTLINE"], ),
            "AlignType":(["Width","Height","Landmarks","JawLine"], ),
            },  
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("Image1","LandmarkImg")
    FUNCTION = "run"

    CATEGORY = _CATEGORY

    def LandMark203_to_68(self,source):
        #jawLine
        out = [source[108]]
        out.append( (source[108+2]*3 + source[108+3] )/4)
        out.append((source[108+4]+source[108+5])/2)
        out.append( (source[108+6] + source[108+7]*3 )/4)
        out.append(source[108+9])
        out.append( (source[108+11]*3 + source[108+12] )/4) 
        out.append((source[108+13]+source[108+14])/2)
        out.append( (source[108+15] + source[108+16]*3 )/4)

        #for i in range(0,7):
        #    out.append((source[110+i*2]+source[111+i*2])/2)  
        out.append(source[126])
        # for i in range(0,7):
        #     out.append((source[128+i*2]+source[129+i*2])/2)  
        out.append( (source[126+2]*3 + source[126+3] )/4)
        out.append((source[126+4]+source[126+5])/2)
        out.append( (source[126+6] + source[126+7]*3 )/4)
        out.append(source[126+9])
        out.append( (source[126+11]*3 + source[126+12] )/4) 
        out.append((source[126+13]+source[126+14])/2)
        out.append( (source[126+15] + source[126+16]*3 )/4)

        out.append(source[144])   
        #leftEyeBow
        out.append(source[145])
        out.append((source[148]+source[162])/2)           
        out.append((source[150]+source[160])/2)   
        out.append((source[152]+source[158])/2)   
        out.append(source[155])
        #rightEyeBow
        out.append(source[165])
        out.append((source[168]+source[182])/2)           
        out.append((source[170]+source[180])/2)   
        out.append((source[172]+source[177])/2)   
        out.append(source[175])

        #nose
        out.append(source[199])
        out.append((source[199]+source[200])/2)           
        out.append(source[200])
        out.append(source[201])
        out.append(source[189])
        out.append(source[190])
        out.append(source[202])
        out.append(source[191])
        out.append(source[192])
        
        #leftEye
        out.append(source[0])
        out.append(source[3])
        out.append(source[8])
        out.append(source[12])
        out.append(source[16])
        out.append(source[21])

        #rightEye
        out.append(source[24])
        out.append(source[28])
        out.append(source[33])
        out.append(source[36])
        out.append(source[39])
        out.append(source[45])

        #UpperLipUp
        out.append(source[48])
        out.append(source[51])
        out.append(source[54])
        out.append(source[57])
        out.append(source[60])
        out.append(source[63])
        out.append(source[66])

        #LowerLipDown
        out.append(source[69])
        out.append(source[72])
        out.append(source[75])
        out.append(source[78])
        out.append(source[81])


        out.append(source[84])
        out.append(source[87])
        out.append(source[90])
        out.append(source[93])
        out.append(source[96])

        out.append(source[99])
        out.append(source[102])
        out.append(source[105])

        return out
    
    def extract_landmarks(self, image):
        """从图像中提取人脸关键点"""
        if self.lmk_extractor is None:
            self.lmk_extractor = LMKExtractor()
        
        # 获取关键点
        face_landmarks = self.lmk_extractor(image)
        if face_landmarks is None or len(face_landmarks) == 0:
            raise ValueError("未在图像中检测到人脸")
        
        # 将关键点转换为numpy数组
        landmarks = []
        for i in range(len(face_landmarks[0])):
            x = face_landmarks[0][i].x * image.shape[1]
            y = face_landmarks[0][i].y * image.shape[0]
            landmarks.append([x, y])
            
        return np.array(landmarks)
        

    def run(self, source_image, target_image, landmarkType, AlignType):
    
            tensor1 = source_image*255
            tensor1 = np.array(tensor1, dtype=np.uint8)
            tensor2 = target_image*255
            tensor2 = np.array(tensor2, dtype=np.uint8)
            output=[]
            image1 = tensor1[0]
            image2 = tensor2[0]

            height, width = image1.shape[:2]
            w = width
            h = height
            
            # 使用mediapipe获取人脸关键点
            landmarks1 = self.extract_landmarks(image1)
            landmarks2 = self.extract_landmarks(image2)
            
            #203个点太多，影响液化算法的运行效率，再次转换成68个点
            use_68_points = True
            if use_68_points:
                landmarks1 = self.LandMark203_to_68(landmarks1)
                landmarks2 = self.LandMark203_to_68(landmarks2)
                landmarks1 = landmarks1[0:65]
                landmarks2 = landmarks2[0:65]

            if use_68_points:
                leftEye1 = np.mean(landmarks1[36:42], axis=0)
                rightEye1 = np.mean(landmarks1[42:48], axis=0)
                leftEye2 = np.mean(landmarks2[36:42], axis=0)
                rightEye2 = np.mean(landmarks2[42:48], axis=0)
                jaw1 = landmarks1[0:17]
                jaw2 = landmarks2[0:17]
                centerOfJaw1 = np.mean(jaw1, axis=0)
                centerOfJaw2 = np.mean(jaw2, axis=0)   
            else:                            
                leftEye1 = np.mean(landmarks1[0:24], axis=0)
                rightEye1 = np.mean(landmarks1[24:48], axis=0)
                leftEye2 = np.mean(landmarks2[0:24], axis=0)
                rightEye2 = np.mean(landmarks2[24:48], axis=0)
                jaw1 = landmarks1[108:145]
                jaw2 = landmarks2[108:145]
                centerOfJaw1 = np.mean(jaw1, axis=0)
                centerOfJaw2 = np.mean(jaw2, axis=0)

            #画面划分成16*16个区域，然后去掉边界框以外的区域。
            src_points = np.array([
                [x, y]
                for x in np.linspace(0, w, 16)
                for y in np.linspace(0, h, 16)
            ])
            
            #上面这些区域同时被加入src和dst，使这些区域不被拉伸（效果是图片边缘不被拉伸）
            src_points = src_points[(src_points[:, 0] <= w/8) | (src_points[:, 0] >= 7*w/8) |  (src_points[:, 1] >= 7*h/8)| (src_points[:, 1] <= h/8)]            
            dst_points = src_points.copy()

            #变形目标人物的landmarks，先计算边界框
            landmarks2 = np.array(landmarks2)
            min_x = np.min(landmarks2[:, 0])
            max_x = np.max(landmarks2[:, 0])
            min_y = np.min(landmarks2[:, 1])
            max_y = np.max(landmarks2[:, 1])
            #得到目标人物的边界框的长宽比
            ratio2 = (max_x - min_x) / (max_y - min_y)
            middlePoint2 = [(max_x + min_x) / 2, (max_y + min_y) / 2]

            #变形原始人物的landmarks，边界框
            landmarks1 = np.array(landmarks1)
            min_x = np.min(landmarks1[:, 0])
            max_x = np.max(landmarks1[:, 0])
            min_y = np.min(landmarks1[:, 1])
            max_y = np.max(landmarks1[:, 1])
            #得到原始人物的边界框的长宽比以及中心点
            ratio1 = (max_x - min_x) / (max_y - min_y)
            middlePoint = [(max_x + min_x) / 2, (max_y + min_y) / 2]
            

            if AlignType=="Width":
            #保持人物脸部边界框中心点不变，垂直方向上缩放，使边界框的比例变得跟目标人物的边界框比例一致    
                if(landmarkType=="ALL"):  
                    dst_points = np.append(dst_points,landmarks1,axis=0)                  
                    target_points = landmarks1.copy()                                        
                else:
                    dst_points = np.append(dst_points,jaw1,axis=0)
                    jaw1=np.array(jaw1)
                    target_points = jaw1.copy() 
                target_points[:, 1] = (target_points[:, 1] - middlePoint[1]) * ratio1 / ratio2 + middlePoint[1]
                src_points = np.append(src_points,target_points,axis=0)#不知道原作者为何把这个数组叫src，其实这是变形后的坐标

            elif AlignType=="Height":
                #保持人物脸部边界框中心点不变，水平方向上缩放，使边界框的比例变得跟目标人物的边界框比例一致
                if(landmarkType=="ALL"):  
                    dst_points = np.append(dst_points,landmarks1,axis=0)    #不知道原作者为何把这个数组叫dst，其实这是变形前的坐标，即原图的坐标              
                    target_points = landmarks1.copy()                                        
                else:
                    dst_points = np.append(dst_points,jaw1,axis=0)#不知道原作者为何把这个数组叫dst，其实这是变形前的坐标，即原图的坐标
                    jaw1=np.array(jaw1)
                    target_points = jaw1.copy() 
                target_points[:, 0] = (target_points[:, 0] - middlePoint[0]) * ratio2 / ratio1 + middlePoint[0]
                src_points = np.append(src_points,target_points,axis=0)#不知道原作者为何把这个数组叫src，其实这是变形后的坐标

            elif AlignType=="Landmarks":
                if(landmarkType=="ALL"):
                    #以双眼中心为基准点，按双眼距离计算缩放系数。效果是变形前后眼睛位置不变
                    MiddleOfEyes1 = (leftEye1+rightEye1)/2
                    MiddleOfEyes2 = (leftEye2+rightEye2)/2
                    distance1 =  ((leftEye1[0] - rightEye1[0]) ** 2 + (leftEye1[1] - rightEye1[1]) ** 2) ** 0.5
                    distance2 =  ((leftEye2[0] - rightEye2[0]) ** 2 + (leftEye2[1] - rightEye2[1]) ** 2) ** 0.5
                    factor = distance1 / distance2
                    MiddleOfEyes2 = np.array(MiddleOfEyes2)
                    target_points = (landmarks2 - MiddleOfEyes2) * factor + MiddleOfEyes1

                    #面部轮廓线则以轮廓线中心点为基准点，缩放系数还是从双眼距离计算
                    centerOfJaw2 = np.array(centerOfJaw2)
                    if use_68_points:
                        jawLineTarget = (landmarks2[0:17] - centerOfJaw2) * factor + centerOfJaw1
                        target_points[0:17] = jawLineTarget
                    else:
                        jawLineTarget = (landmarks2[108:144] - centerOfJaw2) * factor + centerOfJaw1
                        target_points[108:144] = jawLineTarget

                    dst_points = np.append(dst_points,landmarks1,axis=0)#不知道原作者为何把这个数组叫dst，其实这是变形前的坐标，即原图的坐标
                else:
                    #此时只有轮廓线landMark。对齐两个landMark的中心点，然后用2替换掉1
                    dst_points = np.append(dst_points,jaw1,axis=0)#不知道原作者为何把这个数组叫dst，其实这是变形前的坐标，即原图的坐标
                    target_points=(jaw2-centerOfJaw2)+centerOfJaw1
                src_points = np.append(src_points,target_points,axis=0)#不知道原作者为何把这个数组叫src，其实这是变形后的坐标


            elif AlignType=="JawLine":
                lenOfJaw=len(jaw1)
                distance1=  ((jaw1[0][0] - jaw1[lenOfJaw-1][0]) ** 2 + (jaw1[0][1] - jaw1[lenOfJaw-1][1]) ** 2) ** 0.5
                distance2=  ((jaw2[0][0] - jaw2[lenOfJaw-1][0]) ** 2 + (jaw2[0][1] - jaw2[lenOfJaw-1][1]) ** 2) ** 0.5
                factor = distance1 / distance2
                if landmarkType == "ALL":
                    dst_points = np.append(dst_points,landmarks1,axis=0)
                    target_points=(landmarks2-jaw2[0])*factor+jaw1[0]
                    src_points = np.append(src_points,target_points,axis=0)
                else:
                    dst_points = np.append(dst_points,jaw1,axis=0)
                    target_points=(jaw2-jaw2[0])*factor+jaw1[0]
                    src_points = np.append(src_points,target_points,axis=0)
            
            mark_img = draw_pointsOnImg(image1, dst_points, color=(255, 255, 0),radius=4)
            mark_img = draw_pointsOnImg(mark_img, src_points, color=(255, 0, 0),radius=3)
            mark_img = drawLineBetweenPoints(mark_img, dst_points,src_points)
            
            #### 开始对图片进行液化变形
            #Tried many times, finally find out these array should be exchange w,h before go into RBFInterpolator            
            src_points[:, [0, 1]] = src_points[:, [1, 0]]
            dst_points[:, [0, 1]] = dst_points[:, [1, 0]]

            rbfy = RBFInterpolator(src_points,dst_points[:,1],kernel="thin_plate_spline")
            rbfx = RBFInterpolator(src_points,dst_points[:,0],kernel="thin_plate_spline")

            # Create a meshgrid to interpolate over the entire image
            img_grid = np.mgrid[0:height, 0:width]

            # flatten grid so it could be feed into interpolation
            flatten=img_grid.reshape(2, -1).T

            # Interpolate the displacement using the RBF interpolators
            map_y = rbfy(flatten).reshape(height,width).astype(np.float32)
            map_x = rbfx(flatten).reshape(height,width).astype(np.float32)
            # Apply the remapping to the image using OpenCV
            warped_image = cv2.remap(image1, map_y, map_x, cv2.INTER_LINEAR)
            #########  液化变形结束

            warped_image = torch.from_numpy(warped_image.astype(np.float32) / 255.0).unsqueeze(0)               
            mark_img = torch.from_numpy(mark_img.astype(np.float32) / 255.0).unsqueeze(0)  
            output.append(warped_image)
            output.append(mark_img)
    
            return (output)