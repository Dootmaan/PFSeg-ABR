import torch
import glob
import SimpleITK as sitk
import torch.utils.data
import os
from scipy import ndimage
import numpy as np
import cv2
import random
# from config import config

class FinetuneDataset3D(torch.utils.data.Dataset):
    def __init__(self,path,mode='train',augment=False):
        self.data=[]
        self.label=[]
        self.coarse=[]
        self.mode=mode
        self.aug=augment
        images=sorted(glob.glob(os.path.join(path, "cropped_coarse/*_32image.npy")))
        coarse=sorted(glob.glob(os.path.join(path, "cropped_coarse/*_32coarse.npy")))
        labels=sorted(glob.glob(os.path.join(path, "cropped_coarse/*_32mask.npy")))

        # Whether shuffle the dataset. Default not because we need to evaluate test dice for different models.
        # bundle = list(zip(images, coarse, labels))
        # random.shuffle(bundle)
        # images[:], coarse[:], labels[:] = zip(*bundle)

        train_frac, val_frac, test_frac = 0.8, 0.0, 0.2
        n_train = int(train_frac * len(images)) + 1
        n_val = int(val_frac * len(images)) + 1
        n_test = min(len(images) - n_train - n_val, int(test_frac * len(images)))

        print("shape:",np.load(labels[0]).shape)
        
        # Accelarate by loading all data into memory
        if mode=='train':
            print("train:",n_train, "folder:",path)
            images=images[:n_train]
            coarse=coarse[:n_train]
            labels=labels[:n_train]
            for i in range(len(images)):
                print('Adding train sample:',images[i])
                image_arr=np.load(images[i])
                lesion_arr=np.load(labels[i])
                coarse_arr=np.load(coarse[i])
                self.data.append(image_arr)
                self.coarse.append(coarse_arr)
                self.label.append(lesion_arr)
            
        # elif mode=='val':
        #     print("val:", n_val, "folder:",path)
        #     images=images[n_train:n_train+n_val]
        #     labels=labels[n_train:n_train+n_val]
        #     for i in range(len(images)):
        #         print('Adding val sample:',images[i])
        #         image=sitk.ReadImage(images[i])
        #         image_arr=sitk.GetArrayFromImage(image)
        #         lesion=sitk.ReadImage(labels[i])
        #         lesion_arr=sitk.GetArrayFromImage(lesion)
        #         lesion_arr[lesion_arr>1]=1  # 只做WT分割
        #         img,label_seg,label_sr=self.cropMR(image_arr,lesion_arr)
        #         label_seg[label_seg<0.5]=0.
        #         label_seg[label_seg>=0.5]=1.
        #         self.data.append(img)
        #         self.label.append(label_seg)
        #         self.coarse.append(label_sr)
        
        elif mode=='test':
            print("test:", n_test, "folder:",path)
            images=images[n_train:]
            coarse=coarse[n_train:]
            labels=labels[n_train:]

            for i in range(len(images)):
                print('Adding test sample:',images[i])
                image_arr=np.load(images[i])
                lesion_arr=np.load(labels[i])
                coarse_arr=np.load(coarse[i])
                self.data.append(image_arr)
                self.coarse.append(coarse_arr)
                self.label.append(lesion_arr)
            
        else:
            print("Not implemented for this dataset. (No need)")
            raise Exception()

    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):
        if index > self.__len__():
            print("Index exceeds length!")
            return None
            
        return self.data[index],self.coarse[index],self.label[index]

if __name__=='__main__':
    dataset=FinetuneDataset3D('/newdata/why/BraTS20/',mode='train')
    test=dataset.__getitem__(0)
    cv2.imwrite('img.png',test[0][32,:,:]*255)
    cv2.imwrite('seg.png',test[1][32,:,:]*255)
    cv2.imwrite('sr.png',test[2][32,:,:]*255)
    print(test)

