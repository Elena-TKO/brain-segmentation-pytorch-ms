import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_sample, pad_sample, resize_sample, normalize_volume

# import nibabel as nib
# import nilearn as nil
# import scipy.ndimage as ndi




def process_nii_file(path, image = True):
    brain_vol = nib.load(path)
    brain_vol_data_torch = torch.from_numpy(brain_vol.get_fdata())
    brain_vol_data_torch = brain_vol_data_torch.permute(2, 0, 1)
    slices = []
    # print(brain_vol_data_torch.shape)
    for ind in range(len(brain_vol_data_torch)):
        slice_ = brain_vol_data_torch[ind].numpy()
        if image:
            slice_ = np.array([slice_])
            slice_ = torch.from_numpy(slice_)
            slice_ = slice_.permute(1,2,0).numpy()
            slice_ = slice_ / slice_.max() 

            
        # print(slicek_.shape)
        
        slice_ = slice_ * 255
        slices.append(slice_.astype('uint8'))    
    return np.array(slices)


def read_nii_files():
    new_dataset_path = '/mnt/mddp/etrofimenko/segmentation/datasets/files'
    images_nii = []
    masks_nii = []
    for p, d, f in os.walk(new_dataset_path):
        if d == []:
            *_, patient_id = p.split('/')
            for file in f:
                if 'Flair' in file and 'Lesion' not in file:
                    images_nii.append(os.path.join(p, file))
                elif 'Flair' in file and 'Lesion' in file:
                    masks_nii.append(os.path.join(p, file))
    
    
    # print(images_nii[0])
    volumes = {}
    masks = {}
    for index in range(len(images_nii)):
        # if index == 1:
        #     break
        image_nii = images_nii[index]
        mask_nii = masks_nii[index]
        patient_id = image_nii.split('/')[-2]
        
        volumes[patient_id] = process_nii_file(image_nii)
        masks[patient_id] = process_nii_file(mask_nii, False)
        
    return volumes, masks
        
        
        



class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 1
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=10,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            # for filename in sorted(
            #     filter(lambda f: ".tif" in f, filenames),
            #     key=lambda x: int(x.split(".")[-2].split("_")[4]),
            # ):
            for filename in sorted(
                filter(lambda f: ".jpg" in f, filenames),
                key=lambda x: int(x.split(".")[0].split("_")[-1]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    im_ = imread(filepath, as_gray=True)
                    # im_ = np.array([im_]).transpose(1,2,0)
                    mask_slices.append(im_)
                else:
                    im_ = imread(filepath)
                    im_ = np.array([im_]).transpose(1,2,0)
                    # print(im_.shape)
                    image_slices.append(im_)
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])
                # print(np.array(image_slices[1:-1]).shape , np.array(mask_slices[1:-1]).shape)
                # print(np.array(image_slices[1:-1]).max() , np.array(mask_slices[1:-1]).max())
        
        
        # volumes, masks = read_nii_files()
            
        

        self.patients = sorted(volumes)

        print(len(self.patients))
        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        # print("cropping {} volumes...".format(subset))
        # # crop to smallest enclosing volume
        # self.volumes = [crop_sample(v) for v in self.volumes]

        # print("padding {} volumes...".format(subset))
        # # pad to square
        # self.volumes = [pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        # print("normalizing {} volumes...".format(subset))
        # # normalize channel-wise
        # self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        # self.slice_weights = [
        #     (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        # ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                # range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
                range(self.volumes[patient][0].shape[0])
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        # print(image_tensor, mask_tensor)
        return image_tensor, mask_tensor
