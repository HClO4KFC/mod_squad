import torch
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import random
import csv
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import PIL
import skimage  

all_tasks = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_semantic', 'segment_unsup2d', 'segment_unsup25d']
# output[img_type] = rescale_image(output[img_type], new_scale[img_type], current_scale=current_scale[img_type], no_clip=no_clip[img_type])
new_scale, current_scale, no_clip, preprocess = {}, {}, {}, {}

for task in all_tasks:
    new_scale[task], current_scale[task], no_clip[task] = [-1.,1.], None, None
    preprocess[task] = False

# class_object', ' xentropy

# class_scene xentropy

# depth_euclidean l1_loss

# keypoints2d l1
current_scale['keypoints2d'] = [0.0, 0.005]

# keypoints3d

# normal l1_loss

# principal_curvature l2

# reshading l1

# segment_unsup2d metric_loss

preprocess['principal_curvature'] = True

def curvature_preprocess(img, new_dims, interp_order=1):
    img = img[:,:,:2]
    img = img - [123.572, 120.1]
    img = img / [31.922, 21.658]
    return img

def rescale_image(im, new_scale=[-1.,1.], current_scale=None, no_clip=False):
    """
    Rescales an image pixel values to target_scale
    
    Args:
        img: A np.float_32 array, assumed between [0,1]
        new_scale: [min,max] 
        current_scale: If not supplied, it is assumed to be in:
            [0, 1]: if dtype=float
            [0, 2^16]: if dtype=uint
            [0, 255]: if dtype=ubyte
    Returns:
        rescaled_image
    """
    im = skimage.img_as_float(im).astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            im = np.clip(im, min_val, max_val)
        im = im - min_val
        im /= (max_val - min_val) 
    min_val, max_val = new_scale
    im *= (max_val - min_val)
    im += min_val

    return im 

class TaskonomyDataset(data.Dataset):
    
    def __init__(self, img_types, data_dir='/gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium', 
        partition='train', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(TaskonomyDataset, self).__init__()

        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.class_num = {'class_object': 1000, 'class_scene': 365, 'segment_semantic':18}

        def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    scene = row[0]
                    if scene == 'woodbine': # missing from the dataset
                        continue
                    if scene == 'wiconisco': # missing 80 images for edge_texture
                        continue
                    no_list = {'brinnon', 'cauthron', 'cochranton', 'donaldson', 'german',
                        'castor', 'tokeland', 'andover', 'rogue', 'athens', 'broseley', 'tilghmanton', 'winooski', 'rosser', 'arkansaw', 'bonnie', 'willow', 'timberon', 'bohemia', 'micanopy', 'thrall', 'annona', 'byers', 'anaheim', 'duarte', 'wyldwood'
                    }
                    if scene in no_list:
                        continue
                    is_train, is_val, is_test = row[1], row[2], row[3]
                    if is_train=='1' or is_val=='1':
                        label = 'train'
                    else:
                        label = 'test'

                    if label in dictLabels.keys():
                        dictLabels[label].append(scene)
                    else:
                        dictLabels[label] = [scene]
            return dictLabels

        self.data = loadSplit(splitFile = os.path.join(data_dir, 'splits_taskonomy/train_val_test_medium.csv'))
        self.scene_list = self.data[partition]
        self.img_types = img_types
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                                # transforms.Resize(256),
                                # transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.data_list = {}
        for img_type in img_types:
            self.data_list[img_type] = []

        for scene in self.scene_list:
            length = {}
            _max = 0
            for img_type in img_types:
                image_dir = os.path.join(data_dir, img_type, 'taskonomy', scene)
                try:
                    images = sorted(os.listdir(image_dir))
                except:
                    print(scene)
                    continue
                # print(scene, img_type, len(images))
                length[img_type] = len(images)
                _max = max(_max, length[img_type])
                for image in images:
                    self.data_list[img_type].append(os.path.join(image_dir, image))
                    if 'class' in img_type:
                        continue
                    # try:
                    #     img = Image.open(self.data_list[img_type][-1])
                    #     np_img = np.array(img)
                    # except:
                    #     print(self.data_list[img_type][-1])
                    #     # assert False


                    # # if type(np_img.max()) == 'PngImageFile':
                    # if isinstance(np_img.max(), PIL.PngImagePlugin.PngImageFile):
                    #     print(type(np_img.max()), type(np_img), img_type, self.data_list[img_type][index])
                    #     # assert False
                    #     # return output

            for _key, value in length.items():
                if value < _max:
                    # print(_key+'/taskonomy/'+scene, value, _max)
                    print(_key+'/taskonomy/'+scene)

        # assert False
        self.length = len(self.data_list[self.img_types[0]])
        print(len(self.data_list[self.img_types[0]]), self.data_list[self.img_types[0]][self.length-1])
        self._max, self._min = {}, {}
        for img_type in self.img_types:
            self._max[img_type] = -1000000.0
            self._min[img_type] = 100000.0

    def __getitem__(self, index):
        # Load Image
        output = {}
        imgs = []
        no_use = False
        for img_type in self.img_types:
            if img_type == 'class_scene' or img_type == 'class_object':
                target = np.load(self.data_list[img_type][index])
                output[img_type] = torch.from_numpy(target).float()
            else:
                try:
                    img = Image.open(self.data_list[img_type][index])  
                    np_img = np.array(img)  
                except:
                    print(self.data_list[img_type][index])
                    img = Image.open(self.data_list[img_type][index-1])

                imgs.append(img)

        # Transfor operation on image
        if self.resize_scale:
            imgs = [img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR) \
                for img in imgs]

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            imgs = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in imgs]

        if self.fliplr:
            if random.random() < 0.5:
                # imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
                imgs = [img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for img in imgs]

        # imgs = [skimage.img_as_float(img).astype(np.float32) for img in imgs]

        # Value operation on Tensor
        pos = 0
        for img_type in self.img_types:
            if img_type == 'class_scene' or img_type == 'class_object':
                # Note: Seems that only part of class_object is used
                continue
            else:
                output[img_type] = imgs[pos]
                if 'depth' in img_type:
                    output[img_type] = np.array(output[img_type])
                    output[img_type] = np.log(1+output[img_type]) / ( np.log( 2. ** 16.0 ) )
                elif 'curvature' in img_type:
                    output[img_type] = np.array(output[img_type])
                    output[img_type] = curvature_preprocess(output[img_type], (256, 256))
                else:
                    output[img_type] = rescale_image(output[img_type], new_scale[img_type], current_scale=current_scale[img_type], no_clip=no_clip[img_type])
                
                output[img_type] = torch.from_numpy(output[img_type])
                # if 'segment_semantic' in img_type:
                #     continue
                # if 'keypoints2d' in img_type:
                #     output[img_type] = output[img_type] / 4096.0 # I am not quite understand
                # if 'keypoints3d' in img_type:
                #     output[img_type] = output[img_type] / 65536.0
                
                # # output[img_type] = output[img_type].float()
                # if 'edge' in img_type:
                #     output[img_type] = output[img_type] / 12000.0 # I am not quite understand

                

                # # if img_type in max_v:
                # #     output[img_type] = (output[img_type] - min_v[img_type]) * 1.0 / (max_v[img_type] - min_v[img_type])
                
                # if 'segment_unsup2d' in img_type or 'segment_unsup25d' in img_type:
                #     output[img_type] = output[img_type]/255.0

                # if 'rgb' in img_type or 'normal' in img_type or 'principal_curvature' in img_type or 'reshading' in img_type:
                #     output[img_type] = self.transform(output[img_type].permute(2,0,1)/255.0)

                pos = pos + 1

        return output

    def __len__(self):
        return self.length

import random
if __name__ == '__main__':
    # img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'nonfixated_matches', 'normal', 'point_info', 'principal_curvature', 'reshading', 'rgb', 'segment_semantic', 'segment_unsup2d', 'segment_unsup25d']
    # img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_semantic', 'segment_unsup2d', 'segment_unsup25d']
    # img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'normal', 'rgb']
    img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
    # rgb class_object class_scene depth_euclidean depth_zbuffer normal 
    # img_types = ['class_scene', 'class_object', 'rgb', 'normal', 'reshading', 'depth_euclidean', 'segment_unsup2d']
    # img_types = ['rgb', 'class_object', 'normal', 'depth_euclidean', 'keypoints2d']
    A = TaskonomyDataset(img_types, resize_scale=224)
    # # print('done')
    # # A_test = TaskonomyDataset(img_types, resize_scale=256, partition='test')
    # # assert False
    # # print('len: ', len(A))
    # # B = A.__getitem__(32)
    # # # assert False
    for i in tqdm(range(len(A))):
        t = random.randint(0,len(A)-1)
        # print(t)
        B = A.__getitem__(t)
    #     # print(B['edge_texture'][200:230, 200:230]/15000)
    #     # print(B['edge_texture'].min(), B['edge_texture'].max(), B['edge_texture'].mean())
        if i <5:
            for img_type in B.keys():
                print(img_type, B[img_type].min(), B[img_type].max())
        if i > 50:
            break
    #     # print('i: ', i)
    #     # if i==5:
    #     #     for img_type in img_types:
    #     #         print(img_type, B[img_type].min(), B[img_type].max(), B[img_type].shape)
    # assert False
    # train_loader = DataLoader(A, batch_size=32, num_workers=12, shuffle=False, pin_memory=True)
    # for itr, data in tqdm(enumerate(train_loader)):
    #     pass
    #     if itr>400:
    #         break

    # print('min: ', A._min)
    # print('max: ', A._max)

    # for i in range(10):
    #     B = A.__getitem__(i)
    #     # if i==5:
    #     #     for img_type in img_types:
    #     #         print(img_type, B[img_type].min(), B[img_type].max(), B[img_type].shape)
    # print('min: ', A._min)
    # print('max: ', A._max)
    # img_types = ['rgb', 'class_object']

    # assert False
    
    train_set = TaskonomyDataset(img_types, partition='test', resize_scale=298, crop_size=256, fliplr=True)
    print(len(train_set))
    A = train_set.__getitem__(len(train_set)-10)
    # B = train_set.__getitem__(10)
    # for img_type in img_types:
    #     print(B[img_type].shape)
    # assert False
    # # for i in range(1146 * 64, len(train_set)):
    # #     print('i: ', i)
    # #     B = train_set.__getitem__(i)

    train_loader = DataLoader(train_set, batch_size=32, num_workers=30, shuffle=False, pin_memory=True)
    for itr, data in tqdm(enumerate(train_loader)):
        pass
