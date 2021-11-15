from numpy.lib.polynomial import RankWarning
import torch as pt
import numpy as np
from dataset.GuidedBraTSDataset3D import GuidedBraTSDataset3D
from model.PFSeg import PFSeg3D
import cv2
import SimpleITK as sitk

lr=0.0001
epoch=100
batch_size=1
model_path='/path/to/Saved_models'

img_size=(64,96,96)

model=PFSeg3D().cuda()
model.load_state_dict(pt.load(model_path+'/PFSeg_3D_BraTS_patch-free_bs_best.pt',map_location = 'cpu'))

trainset=GuidedBraTSDataset3D('/path/to/BraTS20',mode='all',augment=False)
# valset=BraTSDataset3D('/path/to/BraTS20',mode='val')
# testset=GuidedBraTSDataset3D('/path/to/BraTS20',mode='test')

train_dataset=pt.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=False,drop_last=True)
# val_dataset=pt.utils.data.DataLoader(valset,batch_size=1,shuffle=True,drop_last=True)
# test_dataset=pt.utils.data.DataLoader(testset,batch_size=1,shuffle=True,drop_last=True)

def GenerateCoarseMask():
    model.eval()
    dice_sum=0
    hd_sum=0
    jc_sum=0
    
    for i,data in enumerate(train_dataset):
        output_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
        label_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))

        (inputs,labels,raw_image,guidance,_)=data
        labels3D = pt.autograd.Variable(labels).type(pt.FloatTensor).cuda().unsqueeze(1)
        guidance = pt.autograd.Variable(guidance).type(pt.FloatTensor).cuda().unsqueeze(1)
        
        inputs3D = pt.autograd.Variable(inputs).type(pt.FloatTensor).cuda().unsqueeze(1)
        with pt.no_grad():
            outputs3D,_ = model(inputs3D,guidance)
        outputs3D=np.array(outputs3D.squeeze(0).squeeze(0).cpu().data.numpy())
        output_list=np.zeros((raw_image.shape[1]+64,raw_image.shape[2]+64,raw_image.shape[3]+64))
        output_list[32:-32,32:-32,32:-32]=outputs3D
        label_list=np.zeros((raw_image.shape[1]+64,raw_image.shape[2]+64,raw_image.shape[3]+64))
        label_list[32:-32,32:-32,32:-32]=np.array(labels3D.squeeze(0).squeeze(0).cpu().data.numpy())
        input_real=np.array(raw_image.squeeze(0).numpy())
        input_list=np.zeros((raw_image.shape[1]+64,raw_image.shape[2]+64,raw_image.shape[3]+64))
        input_list[32:-32,32:-32,32:-32]=input_real

        output_list[output_list<0.5]=0.
        output_list[output_list>=0.5]=1.

        results=np.where(output_list!=0)
        x_list=results[0]
        y_list=results[1]
        z_list=results[2]

        x_max=x_list.max()
        x_min=x_list.min()
        y_max=y_list.max()
        y_min=y_list.min()
        z_max=z_list.max()
        z_min=z_list.min()

        x_length=64*(1+(x_max-x_min)//64)  #确保是16的倍数
        y_length=64*(1+(y_max-y_min)//64)
        z_length=64*(1+(z_max-z_min)//64)

        x_center=(x_max-x_min)//2+x_min
        y_center=(y_max-y_min)//2+y_min
        z_center=(z_max-z_min)//2+z_min

        bbox_xmin=x_center-x_length//2
        bbox_xmax=x_center+x_length//2
        bbox_ymin=y_center-y_length//2
        bbox_ymax=y_center+y_length//2
        bbox_zmin=z_center-z_length//2
        bbox_zmax=z_center+z_length//2
        
        # cropped_coarse=np.zeros((x_length,y_length,z_length))
        # cropped_image=np.zeros((x_length,y_length,z_length))
        # cropped_mask=np.zeros((x_length,y_length,z_length))

        cropped_image=input_list[bbox_xmin:bbox_xmax,bbox_ymin:bbox_ymax,bbox_zmin:bbox_zmax]
        cropped_coarse=output_list[bbox_xmin:bbox_xmax,bbox_ymin:bbox_ymax,bbox_zmin:bbox_zmax]
        cropped_mask=label_list[bbox_xmin:bbox_xmax,bbox_ymin:bbox_ymax,bbox_zmin:bbox_zmax]

        if not(cropped_mask.shape==cropped_image.shape):
            raise Exception()
        if not(cropped_image.shape[0]%16==0 and cropped_image.shape[1]%16==0 and cropped_image.shape[2]%16==0):
            raise Exception()

        # save the cropped images for next round training
        np.save('/path/to/BraTS20/cropped_coarse/Case_{:3d}_64image.npy'.format(i+1),cropped_image)
        np.save('/path/to/BraTS20/cropped_coarse/Case_{:3d}_64coarse.npy'.format(i+1),cropped_coarse)
        np.save('/path/to/BraTS20/cropped_coarse/Case_{:3d}_64mask.npy'.format(i+1),cropped_mask)

        # final_img=np.zeros(shape=(2*img_size[1],2*2*img_size[2]))
        # final_img[:,:2*img_size[2]]=output_list[0,0,64,:,:]*255
        # final_img[:,2*img_size[2]:]=label_list[0,0,64,:,:]*255
        # cv2.imwrite('TestPhase_BraTS.png',final_img)
        
        pr_sum = output_list.sum()
        gt_sum = label_list.sum()
        pr_gt_sum = np.sum(output_list[label_list == 1])
        dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        dice_sum += dice
        print("dice:",dice)

        # hausdorff=hd95(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))
        # jaccard=jc(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))

        # hd_sum+=hausdorff 
        # jc_sum+=jaccard

    print("Finished. Total dice: ",dice_sum/len(train_dataset),'\n')
    print("Finished. Avg Jaccard: ",jc_sum/len(train_dataset))
    print("Finished. Avg hausdorff: ",hd_sum/len(train_dataset)) 
    return dice_sum/len(train_dataset)

GenerateCoarseMask()