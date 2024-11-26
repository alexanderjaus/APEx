import os
from os.path import join

import nibabel as nib
from nibabel import processing

import pandas as pd
import numpy as np

import shutil

from datetime import datetime
import os
from os.path import join
import argparse
import nibabel as nib
import pandas as pd
import numpy as np
import shutil
from datetime import datetime
from gzip import BadGzipFile

import cv2

DEBUG = True

def create_dataset_base_folder(autopet_path, atlas_labels, pet_atlas_complete,metadata_csv_path):
    
    autopet_subjects = [x for x in os.listdir(autopet_path) if x.startswith("PETCT")]
    metadata = pd.read_csv(metadata_csv_path)
    metadata["study_date_parsed"] = pd.to_datetime(metadata["Study Date"])

    for subject in autopet_subjects[:200]: # DEBUG
        full_path = join(autopet_path,subject)
        studies = os.listdir(full_path)
        for study in studies:
            
            #Get date of the study of the current subject in Autopet Folder
            month, day, year = study.split("-")[:3]
            
            year = int(year)
            month = int(month)
            day = int(day)
            
            study_date = datetime(year=year, month=month, day=day)
            
            #Look up in metadata to identify correct atlas files
            metadata_rows_mask = metadata["Subject ID"]==subject
            date_based_mask = metadata["study_date_parsed"] == study_date
            
            study_uids = metadata[np.logical_and(metadata_rows_mask, date_based_mask)]["Study UID"]
            assert study_uids.unique().shape[0] == 1, "Expected only a single matching study uid"
            study_uid = study_uids.iloc[0]
            
            subject_id = subject.split("_")[1]
            
            atlas_file_name = f"AutoPET_{subject_id}_{study_uid[-5:]}.nii.gz"
            
            #Check if the atlas file actually exists
            print(atlas_file_name)
            if os.path.isfile(join(atlas_labels,atlas_file_name)):

                # Now copy everything together 

                #1) Create unique folder in target directory
                target_folder = f"AutoPET_{subject}_{year}_{month}_{day}"
                target_path = join(join(pet_atlas_complete,target_folder))
                if not os.path.isdir(target_path):
                    print(f"Making directory {target_path}")
                    os.mkdir(target_path)

                # Copy all files into the new directoy
                for file in os.listdir(join(autopet_path,subject,study)):
                    print(f"{join(autopet_path,subject,study,file)}--->{target_path}")
                    shutil.copy2(join(autopet_path,subject,study,file),target_path)

                #Copy the atlas file
                shutil.copy2(join(atlas_labels, atlas_file_name), target_path)
                print(f"{join(atlas_labels, atlas_file_name)}--->{target_path}")
            else:
                print("Atlas file not found, skip this patient")
    
def resample_anatomy_labels(petct_atlas):
    file_list = os.listdir(petct_atlas)
    for idx, folder in enumerate(file_list):
        print(f"Processing file {folder}. {idx}/{len(file_list)}")
        atlas_files = [x for x in os.listdir(join(petct_atlas,folder)) if x.startswith("AutoPET")]
        assert len(atlas_files) == 1, "Expected only one Atlas file"
        atlas_file = nib.load(join(petct_atlas, folder,atlas_files[0]))
        target_file= nib.load(join(petct_atlas, folder, "CTres.nii.gz"))
        resampled_atlas = processing.resample_from_to(atlas_file,target_file,order=0)
        
        assert np.max(np.unique(resampled_atlas.get_fdata()))<=145, "Got a strange max value"
        #Compare shapes
        assert resampled_atlas.shape == target_file.shape, f"Expect same shape, but got for resampled atlas {resampled_atlas.shape} which should be {target_file.shape}" 
        nib.save(resampled_atlas, join(petct_atlas,folder,"atlas_resampled.nii.gz"))

def sliceup_volumes(volumes_dataset, slices_dataset_target):
    def load_file(path):
        try:
            file = nib.load(path)
            file_grid = file.get_fdata()
            return file_grid
        except BadGzipFile:
            print("#######WARNING#########")
            print(f"Corrupted file {path}")
            print("#######WARNING#########")
            return False
        
    #Compute foreground mean and std for CTres
    suv_fg_mean = 6.0768
    suv_fg_std = 4.9945
    epsilon = 1e-6 # Add for numerical stability
    
    f_img_save = lambda name, img: cv2.imwrite(name,img) 
    
    if not os.path.exists(slices_dataset_target):
        os.mkdir(slices_dataset_target)
    skipped_studies = []
    for study in os.listdir(volumes_dataset):
        print(f"Processing file {study}")
        source_path = join(volumes_dataset,study)
        target_study_path = join(slices_dataset_target,study)
        
        #Create a folder for the target study
        if not os.path.isdir(target_study_path):
            os.mkdir(target_study_path)
        
        for file_name in os.listdir(source_path):
            print(f"Processing file {file_name}")
            file_target_path = join(target_study_path, file_name.split(".")[0] + "_slices")
        
            # Differentiate between the files
            if file_name == "CTres.nii.gz":
                #Create a folder for each file in target directory
                if not os.path.isdir(file_target_path):
                    os.mkdir(file_target_path)
                
                #Load file
                file_grid = load_file(join(source_path,file_name))
                if type(file_grid) is bool:
                    print("Skipping this study")
                    skipped_studies.append(study)
                    break
                foreground_cutoff = -800
                foreground = file_grid[file_grid>foreground_cutoff]
                clip_lower, clip_higher = np.quantile(foreground,(0.05,0.995))

                
                #Clip foreground to caculate foreground mean 
                foreground[foreground<clip_lower] = clip_lower
                foreground[foreground>clip_higher] = clip_higher
                foreground_mean = np.mean(foreground)
                foreground_std = np.std(foreground)
                
                
                #Normalize standard image according to foreground mean and std
                file_grid[file_grid<clip_lower] = clip_lower
                file_grid[file_grid>clip_higher] = clip_higher
                file_grid = (file_grid - foreground_mean)/foreground_std
                for slice_idx in range(file_grid.shape[-1]):
                    twodim_slice = file_grid[:,:,slice_idx].astype(np.float32)
                    file_name = f"{study}_ctres_{slice_idx}.tif"
                    twodim_slice_norm = (twodim_slice - np.min(twodim_slice))/(np.max(twodim_slice)-np.min(twodim_slice)+epsilon)
                    f_img_save(join(file_target_path,file_name),twodim_slice_norm)
            
            elif file_name == "SUV.nii.gz":
                if not os.path.isdir(file_target_path):
                    os.mkdir(file_target_path)
                
                
                file_grid = load_file(join(source_path,file_name))
                if type(file_grid) is bool:
                    print("Skipping this study")
                    skipped_studies.append(study)
                    break

                #Debrain the pet
                this_label = nib.load(join(source_path,"atlas_resampled.nii.gz"))
                label_grid = this_label.get_fdata()
                brain_label = 25
                brain_mask = label_grid == brain_label

                file_grid[brain_mask] = 0
                
                #clip_higher = np.quantile(file_grid.flatten())
                #file_grid[file_grid<clip_lower] = clip_lower
                #file_grid[file_grid>clip_higher] = clip_higher
                mean = suv_fg_mean
                std = suv_fg_std
                file_grid = (file_grid-mean)/std

                for slice_idx in range(file_grid.shape[-1]):
                    twodim_slice = file_grid[:,:,slice_idx].astype(np.float32)
                    file_name = f"{study}_suv_{slice_idx}.tif"
                    twodim_slice_norm = (twodim_slice - np.min(twodim_slice))/(np.max(twodim_slice)-np.min(twodim_slice)+epsilon)
                    f_img_save(join(file_target_path,file_name),twodim_slice_norm)
            
            elif file_name == "atlas_resampled.nii.gz":
                if not os.path.isdir(file_target_path):
                    os.mkdir(file_target_path)

                file_grid = load_file(join(source_path,file_name))
                if type(file_grid) is bool:
                    print("Skipping this study")
                    skipped_studies.append(study)
                    break
                
                for slice_idx in range(file_grid.shape[-1]):
                    twodim_slice = file_grid[:,:,slice_idx].astype(np.uint8)
                    file_name = f"{study}_atlaslabels_{slice_idx}.tif"
                    f_img_save(join(file_target_path,file_name),twodim_slice)

            elif file_name == "SEG.nii.gz":
                if not os.path.isdir(file_target_path):
                    os.mkdir(file_target_path)
                file_grid = load_file(join(source_path,file_name))
                if type(file_grid) is bool:
                    print("Skipping this study")
                    skipped_studies.append(study)
                    break
                for slice_idx in range(file_grid.shape[-1]):
                    twodim_slice = file_grid[:,:,slice_idx].astype(np.uint8)
                    file_name = f"{study}_seglabel_{slice_idx}.tif"
                    f_img_save(join(file_target_path,file_name),twodim_slice)

def pool_the_data(slices_dataset_target, cleanup=False):
    """Generate three folders: images, annotations_anatomy, annotations_pathology if they do not exist"""
    if not os.path.exists(join(slices_dataset_target,"images")):
        os.mkdir(join(slices_dataset_target,"images"))
    if not os.path.exists(join(slices_dataset_target,"annotations_anatomy")):
        os.mkdir(join(slices_dataset_target,"annotations_anatomy"))
    if not os.path.exists(join(slices_dataset_target,"annotations_pathology")):
        os.mkdir(join(slices_dataset_target,"annotations_pathology"))
    
    #Pool the data
    for study in filter(lambda x: x not in ["images", "annotations_anatomy", "annotations_pathology"], os.listdir(slices_dataset_target)):
        study_path = join(slices_dataset_target,study)
        for data_type in os.listdir(study_path):
            data_type_path = join(study_path,data_type)
            for file in os.listdir(data_type_path):
                if data_type == "atlas_resampled_slices":
                    shutil.copy2(join(data_type_path,file), join(slices_dataset_target, "annotations_anatomy", file))
                elif data_type == "CTres_slices":
                    shutil.copy2(join(data_type_path,file), join(slices_dataset_target, "images", file))
                elif data_type == "SEG_slices":
                    shutil.copy2(join(data_type_path,file), join(slices_dataset_target, "annotations_pathology", file))
                elif data_type == "SUV_slices":
                    shutil.copy2(join(data_type_path,file), join(slices_dataset_target, "images", file))
                else:
                    raise ValueError(f"Unknown data type {data_type}")
    

    if cleanup:
        for study in filter(lambda x: x not in ["images", "annotations_anatomy", "annotations_pathology"], os.listdir(slices_dataset_target)):
            shutil.rmtree(join(slices_dataset_target,study))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("autopet_path", help="Path to AutoPET directory")
    parser.add_argument("atlas_labels", help="Path to atlas labels directory")
    parser.add_argument("pet_atlas_complete", help="Path to directory to create volumetric Apex dataset")
    parser.add_argument("slices_dataset_target", help="Path to directory to create 2D slices dataset")
    parser.add_argument("--cleanup", help="Remove the original files after pooling", action="store_true", default=True)
    
    parser.add_argument("metadata_csv_path", help="Path to metadata CSV file, present in AutoPET directory")
    args = parser.parse_args()

    #Copy all relevant files into the new directory
    create_dataset_base_folder(args.autopet_path, args.atlas_labels, args.pet_atlas_complete, args.metadata_csv_path)
    
    #Resample the anatomy labels
    resample_anatomy_labels(args.pet_atlas_complete)
    
    #Sliceup volumetric images to 2D slices in the same directory
    sliceup_volumes(args.pet_atlas_complete,args.slices_dataset_target)

    #Pool all the data together
    pool_the_data(args.slices_dataset_target, args.cleanup)




