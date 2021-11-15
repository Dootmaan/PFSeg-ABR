# PFSeg-ABR
Patch-Free 3D Segmentation with Adaptive Boundary Refinement

This repo is an extension of [PFSeg](https://github.com/Dootmaan/PFSeg). We added a boundary refinement stage to the coarse output of PFSeg. It uses the original HR image patch cropped from the location of the generated mask, and the refinement is conducted on the basis of the output. To keep the spacing information, we designed a dynamic extending strategy which adaptively calculate the number of voxels needed to be extended. Since the computational cost of ABR varies with different datasets and voxel extending strategy, it is espcially recommended for small objects segmentation. ~~More details and codes will be released upon paper publication.~~

We have changed our research goal and decide not to focus on ABR. The code still will be released. ABR actually works for all the patch-free or patch-based methods and can be viewed as a finetuning precedure. We are sorry that this repo may not receive further updates.

We did some rough format changing before realeasing the code. The modified code has NOT been tested so it may has potential errors. Feel free to ask us any questions in the issues, and we will try our best to answer. Not that the default number of ABR extending voxels is 64 currently, which is fine for BRATS2020 but may cause OOM for other datasets.

## 1. Prepare your dataset
Please follow the instructions on http://braintumorsegmentation.org/ to get your copy of the BRATS2020 dataset. 

## 2. Install dependencies
Our code should work with Python>=3.5 and PyTorch >=0.4.1. 

Please make sure you also have the following libraries installed on your machine:
- PyTorch
- NumPy
- MedPy
- tqdm

Optional libraries (they are only needed when run with the -v flag):
- opencv-python

## 3. Run the code
Firstly, clone our code by running

```
    git clone git@github.com:Dootmaan/PFSeg-ABR.git
```
Normally you only have to specify the path to BRATS2020 dataset to run the code.

For example you can use the following command (of course you need to change directory to ./PFSeg first):

```
    CUDA_VISIBLE_DEVICES=0 nohup python3 -u twoStageTest.py -dataset_path "/home/somebody/BRATS2020/" > train_PFSeg.log 2>&1 &
```

You can also add the -v flag to have verbose output. Our code supports multi-gpu environment and all you have to do is specifying the indices of the available GPUs. Directly running train_PFSeg.py will use all the GPUs on your machine and use the default parameter settings. **The minimun requirements for running our code is a single GPU with 11G video memory.**

Click [here](https://drive.google.com/file/d/1kG2kYU_56-0UV2E2I59c1qYphoYRdziK/view?usp=sharing) to download the pretrained PFS weights for our framework. click [here]() to download the pretrained weights for the ABR model (voxel extending dividor being 64).
