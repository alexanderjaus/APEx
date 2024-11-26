import os
import json
from PIL import Image


from functools import partial

# Process the files and create COCO annotations
def register_dataset(split_files, anatomy_annotations, pathology_annotations, images_dir):
    image_id = 1
    #annotation_id = 1
    dataset_list = []
    for root, _, files in os.walk(pathology_annotations):
        for file in files:
            #TODO: Make this two separate dataset registration functions
            
            base_name = '_'.join(file.split('_')[:-2])
            if base_name in split_files:
                # Image details
                
                ct_path = os.path.join(images_dir, f"{base_name}_ctres_{file.split('_')[-1]}")
                pet_path = os.path.join(images_dir, f"{base_name}_suv_{file.split('_')[-1]}")
                
                anatomy_path = os.path.join(anatomy_annotations, f"{base_name}_atlaslabels_{file.split('_')[-1]}")
                pathology_path = os.path.join(pathology_annotations, f"{base_name}_seglabel_{file.split('_')[-1]}")

                assert os.path.exists(ct_path), f"CT path {ct_path} does not exist"
                assert os.path.exists(pet_path), f"Pet path {pet_path} does not exist"  
                assert os.path.exists(anatomy_path), f"Anatomy path {anatomy_path} does not exist"
                assert os.path.exists(pathology_path), f"Pathology path {pathology_path} does not exist"

                width, height = 400, 400
                dataset_list.append({
                    "id": image_id,
                    "file_name": ct_path,
                    "pet_path": pet_path,
                    "anatomy_path": anatomy_path,
                    "pathology_path": pathology_path,
                    "sem_seg_file_name": pathology_path, # Needed due to detectron2 evaluate requirements
                    "width": width,
                    "height": height,
                })

                image_id += 1
    return dataset_list

def get_dataset_registration(split_file, anatomy_annotations, pathology_annotations, images_dir, split="train"):
    with open(split_file, 'r') as f:
        splits = json.load(f)
    assert split in ["train","test","validation","val"], f"Expected split to be in [train,test,validation] but got {split}"
    this_registration_method = partial(register_dataset, splits[split], anatomy_annotations, pathology_annotations, images_dir)
    return this_registration_method

def register_large_ts():
    image_id = 1
    source_dir = "/lsdf/users/ajaus/new_large_testset_sliced_petctmix"
    image_folder = "images"
    annotation_folder = "annotations"
    dataset_list = []
    for root, _, files in os.walk(os.path.join(source_dir,image_folder)):
        for file in files:
            base_name = '_'.join(file.split('_')[:-1])
            image_path = os.path.join(source_dir, image_folder, file)
            annotation_path = os.path.join(source_dir,annotation_folder,file)
            assert os.path.exists(annotation_path), f"Annotation path {annotation_path} does not exist"
            width, height = 400, 400
            dataset_list.append({
                "id": image_id,
                "file_name": image_path,
                "width": width,
                "height": height,
                "sem_seg_file_name": annotation_path
            })

            image_id += 1
    return dataset_list


if __name__ == '__main__':
    
    reg_function = get_dataset_registration(split_file="/local/cv_split_1.json", annotation_dir="/local/detectron_ct_data/annotations", images_dir="/local/detectron_ct_data/images")
    dataset_list = reg_function()
    print("Registered")
