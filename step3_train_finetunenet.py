import torch as pt
import numpy as np
from model.FinetuneNet import FinetuneNet
from dataset.FinetuneDataset3D import FinetuneDataset3D
import cv2
from loss.DiceLoss import BinaryDiceLoss

lr=0.0001
epoch=80
batch_size=1
model_path='/path/to/Saved_models'

# trainset=BraTSDataset3D('/path/to/BraTS20',train=True)
# testset=BraTSDataset3D('/path/to/BraTS20',train=False)

trainset=FinetuneDataset3D('/path/to/BraTS20/',mode='train',augment=False)
# valset=FinetuneDataset3D('/path/to/BraTS20/cropped_coarse/',mode='val')
testset=FinetuneDataset3D('/path/to/BraTS20/',mode='test')

train_dataset=pt.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,drop_last=True)
# val_dataset=pt.utils.data.DataLoader(valset,batch_size=1,shuffle=True,drop_last=True)
test_dataset=pt.utils.data.DataLoader(testset,batch_size=1,shuffle=True,drop_last=True)

# train_dataset=pt.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,drop_last=True)
# test_dataset=pt.utils.data.DataLoader(testset,batch_size=1,shuffle=True,drop_last=True)
# allset=BraTSDataset3D('/path/to/BraTS20',mode='all')
# all_dataset=pt.utils.data.DataLoader(allset,batch_size=1,shuffle=False,drop_last=True)
# train_dataset=[]
# val_dataset=[]

model=FinetuneNet(in_channels=2,out_channels=1).cuda()
# model.load_state_dict(pt.load(model_path+'/FinetuneNet_3D_BraTS_patch-free48_bs1_best.pt',map_location = 'cpu'))

lossfunc_sr=pt.nn.MSELoss()
lossfunc_seg=pt.nn.BCELoss()
lossfunc_dice=BinaryDiceLoss()
# lossfunc_fa=FALoss3D()
optimizer = pt.optim.Adam(model.parameters(), lr=lr)
# scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
scheduler=pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=10)

# def ValModel():
#     model.eval()
#     dice_sum=0
#     weight_map=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
#     for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
#         for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
#             for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
#                 weight_map[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=1
    
#     weight_map[weight_map==0]=1
#     weight_map=1./weight_map
#     for i,data in enumerate(val_dataset):
#         output_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
#         label_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))

#         (_,labels,inputs)=data   # use raw label_sr as input
#         labels3D = pt.autograd.Variable(labels).type(pt.FloatTensor).cuda().unsqueeze(1)
        
#         for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
#             for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
#                 for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
#                     inputs3D = pt.autograd.Variable(inputs[:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]).type(pt.FloatTensor).cuda().unsqueeze(1)
#                     with pt.no_grad():
#                         outputs3D = model(inputs3D)
#                     outputs3D=np.array(outputs3D.cpu().data.numpy())
#                     # outputs3D=ndimage.interpolation.zoom(outputs3D,[1,1,2,2,2],order=3)
#                     # outputs3D[outputs3D<0.5]=0
#                     # outputs3D[outputs3D>=0.5]=1
#                     output_list[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=outputs3D

#         label_list=np.array(labels3D.cpu().data.numpy())

#         output_list=np.array(output_list)*weight_map

#         # label_list=np.array(label_list)

#         output_list[output_list<0.5]=0
#         output_list[output_list>=0.5]=1

#         # final_img=np.zeros(shape=(2*img_size[1],2*2*img_size[2]))
#         # final_img[:,:2*img_size[2]]=output_list[0,0,64,:,:]*255
#         # final_img[:,2*img_size[2]:]=label_list[0,0,64,:,:]*255
#         # cv2.imwrite('TestPhase_Res_patchfree_BraTS.png',final_img)

#         pr_sum = output_list.sum()
#         gt_sum = label_list.sum()
#         pr_gt_sum = np.sum(output_list[label_list == 1])
#         dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
#         dice_sum += dice
#         print("dice:",dice)

#         output_list=[]
#         label_list=[]

#     print("Finished. Total dice: ",dice_sum/len(val_dataset),'\n')
#     return dice_sum/len(val_dataset)


def TestModel():
    model.eval()
    dice_sum=0
    
    for i,data in enumerate(test_dataset):

        (inputs,coarse,labels)=data   # use raw label_sr as input
        labels3D = pt.autograd.Variable(labels).type(pt.FloatTensor).cuda().unsqueeze(1)
        
        inputs3D = pt.autograd.Variable(inputs).type(pt.FloatTensor).cuda().unsqueeze(1)
        coarse3D = pt.autograd.Variable(coarse).type(pt.FloatTensor).cuda().unsqueeze(1)
        with pt.no_grad():
            outputs3D = model(pt.cat((inputs3D,coarse3D),dim=1))
        outputs3D=np.array(outputs3D.cpu().data.numpy())
        # outputs3D=ndimage.interpolation.zoom(outputs3D,[1,1,2,2,2],order=3)
        # outputs3D[outputs3D<0.5]=0
        # outputs3D[outputs3D>=0.5]=1
        output_list=outputs3D

        label_list=np.array(labels3D.cpu().data.numpy())

        output_list=np.array(output_list)

        # label_list=np.array(label_list)

        output_list[output_list<0.5]=0
        output_list[output_list>=0.5]=1

        # final_img=np.zeros(shape=(2*img_size[1],2*2*img_size[2]))
        # final_img[:,:2*img_size[2]]=output_list[0,0,64,:,:]*255
        # final_img[:,2*img_size[2]:]=label_list[0,0,64,:,:]*255
        # cv2.imwrite('TestPhase_Res_patchfree_BraTS.png',final_img)

        pr_sum = output_list.sum()
        gt_sum = label_list.sum()
        pr_gt_sum = np.sum(output_list[label_list == 1])
        dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        dice_sum += dice
        print("dice:",dice)

        output_list=[]
        label_list=[]

    print("Finished. Total dice: ",dice_sum/len(test_dataset),'\n')
    return dice_sum/len(test_dataset)


# best_dice_sum=0
# data_induce = np.arange(0, allset.__len__())
# kf = KFold(n_splits=5)
# fold=1
# for train_index, val_index in kf.split(data_induce):
#     model=ResUNet3D(in_channels=1,out_channels=1).cuda()
#     print('Fold',fold,'start')
#     train_subset = pt.utils.data.dataset.Subset(allset, train_index)
#     val_subset = pt.utils.data.dataset.Subset(allset, val_index)
#     train_dataset = pt.utils.data.DataLoader(train_subset,batch_size=1,shuffle=False,drop_last=True)
#     val_dataset = pt.utils.data.DataLoader(val_subset,batch_size=1,shuffle=False,drop_last=True)

#     optimizer = pt.optim.Adam(model.parameters(), lr=lr)
#     scheduler=pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=20)
#     # scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

best_dice=0
for x in range(epoch):
    model.train()
    loss_sum=0
    print('==>Epoch',x,': lr=',optimizer.param_groups[0]['lr'],'==>\n')

    for i,data in enumerate(train_dataset):
        (image,coarse,labels)=data
        optimizer.zero_grad()
        image = pt.autograd.Variable(image).type(pt.FloatTensor).cuda().unsqueeze(1)
        coarse = pt.autograd.Variable(coarse).type(pt.FloatTensor).cuda().unsqueeze(1)
        labels = pt.autograd.Variable(labels).type(pt.FloatTensor).cuda().unsqueeze(1)
        outputs_seg = model(pt.cat((image,coarse),dim=1))
        loss_seg = lossfunc_seg(outputs_seg, labels)+lossfunc_dice(outputs_seg,labels)

        loss_seg.backward()
        optimizer.step()

        # loss_sum+=loss_seg.item()

        if i%10==0:
        #     final_img=np.zeros(shape=(size,size*3))
            print('[epoch {:3d},iter {:5d}]'.format(x,i),'loss:',loss_seg.item())
            cv2.imwrite('image.png',image.cpu().data.numpy()[0,0,image.shape[2]//2,:,:]*255)
            cv2.imwrite('coarse.png',coarse.cpu().data.numpy()[0,0,image.shape[2]//2,:,:]*255)
            cv2.imwrite('mask.png',labels.cpu().data.numpy()[0,0,image.shape[2]//2,:,:]*255)
        #     final_img[:,0:size]=outputs_seg.cpu().data.numpy()[0,0,crop_size[0],:,:]*255
        #     # final_img[:,128:256]=outputs_sr.cpu().data.numpy()[0,0,31,:,:]*255
        #     final_img[:,size:(2*size)]=labels.cpu().data.numpy()[0,0,crop_size[0],:,:]*255
        #     # final_img[:,384:512]=labels_sr.cpu().data.numpy()[0,0,31,:,:]*255
        #     final_img[:,(2*size):]=image.cpu().data.numpy()[0,0,crop_size[0],:,:]*255
        #     cv2.imwrite('resunet_3d_patchfree_combine.png',final_img)

    # scheduler.step()

    print('==>End of epoch',x,'==>\n')

    print('===VAL===>')
    dice=TestModel()
    scheduler.step(dice)
    if dice>best_dice:
        best_dice=dice
        print('New best dice! Model saved to',model_path+'/FinetuneNet_3D_BraTS_patch-free_bs'+str(batch_size)+'_best.pt')
        pt.save(model.state_dict(), model_path+'/FinetuneNet_3D_BraTS_patch-free_bs'+str(batch_size)+'_best.pt')
    # print('===TEST===>')
    # TestModel() 
# print('Fold',fold,'best', best_dice)
# best_dice_sum+=best_dice
# fold+=1

print('\nBest Dice:',best_dice)