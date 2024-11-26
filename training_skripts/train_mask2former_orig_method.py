# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Apex Training Script.

This script is a Modification of the Mask2Former Training Skript.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set, Optional, Union

import torch
import cv2
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)

from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from detectron2.data import DatasetCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

from detectron2.structures import BitMasks, Instances

from detectron2.config import configurable
from detectron2.modeling import build_model


import sys
sys.path.append('prepare_data')
sys.path.append("model")


from prepare_data.register_segmentation_datasets import get_dataset_registration

from labels import full_labels, anatomical_labels, pathological_labels

from apex import (
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

def add_loss_weights_to_cfg(cfg):
    cfg.MODEL.SEM_SEG_HEAD.WEIGH_CLASSES = "cfg.MODEL.SEM_SEG_HEAD.WEIGH_CLASSES"
    cfg.MODEL.SEM_SEG_HEAD.INVERSE_SCALE = True

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs

def prepare_dataset(args):
    
    if "split" in args.opts:
        option_idx = args.opts.index("split")
        args.opts.pop(option_idx)
        split = args.opts.pop(option_idx)
        split = int(split)
        assert 1<=split<=5, f"Expect the split to be in the range of 1 - 5, but got {split}"
        split_file_location = f"training_skripts/cv_split_{split}.json"
    else:
        split_file_location = "training_skripts/cv_split_1.json"
    
    if "DETECTRON_DATASET" not in os.environ:
        raise ValueError("Please set the environment variable 'DETECTRON_DATASET' to the path of the datasets")
    base_dir = os.environ["DETECTRON_DATASET"]
    
    annotation_dir_anatomy = os.path.join(base_dir, "annotations_anatomy")
    annotation_dir_pathology = os.path.join(base_dir, "annotations_pathology")
    
    images_dir_location = os.path.join(base_dir, "images")

    
    ### Register the full dataset containing labels for anatomy and pathology
    registration_function_train_full_dataset = get_dataset_registration(split_file=split_file_location, 
                                                           anatomy_annotations=annotation_dir_anatomy, 
                                                           pathology_annotations=annotation_dir_pathology,
                                                           images_dir=images_dir_location,
                                                           split="train"
                                                           )
    registration_function_val_full_dataset = get_dataset_registration(split_file=split_file_location, 
                                                           anatomy_annotations=annotation_dir_anatomy, 
                                                           pathology_annotations=annotation_dir_pathology,
                                                           images_dir=images_dir_location,
                                                           split="validation"
                                                           )

    DatasetCatalog.register("petctmix_dataset_train_full", registration_function_train_full_dataset)
    print("Registered the training dataset")
    
    DatasetCatalog.register("petctmix_dataset_val_full", registration_function_val_full_dataset)
    print("Registered validation dataset")
    
    MetadataCatalog.get("petctmix_dataset_train_full").set(stuff_classes=pathological_labels)
    MetadataCatalog.get("petctmix_dataset_train_full").set(evaluator_type="sem_seg")
    MetadataCatalog.get("petctmix_dataset_train_full").set(ignore_label=None)

    MetadataCatalog.get("petctmix_dataset_val_full").set(stuff_classes=pathological_labels)
    MetadataCatalog.get("petctmix_dataset_val_full").set(evaluator_type="sem_seg")
    MetadataCatalog.get("petctmix_dataset_val_full").set(ignore_label=None)
    print("Dataset and Metadata registered")
    

class CustomM2FMapper(MaskFormerSemanticDatasetMapper):

    def __init__(self, cfg, is_train=True, augmentations=[], load_with_numpy=False, load_pathology_separately=False, load_with_pil = False, dataype="petct"):
            super().__init__(cfg, is_train, augmentations=augmentations)
            self.augmentations = T.AugmentationList(augmentations)
            self.load_with_numpy = load_with_numpy
            self.load_with_pil = load_with_pil
            self.dataype = dataype
            self.load_pathology_separately = load_pathology_separately
            if self.load_pathology_separately:
                if "DETECTRON_DATASET" in os.environ:
                    self.data_base_dir = os.environ["DETECTRON_DATASET"]
                else:
                    self.data_base_dir = "/home/hk-project-cvhcimed/yu2513/Atlas_workspace/"
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  
        

        ct = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_UNCHANGED)
        pet = cv2.imread(dataset_dict["pet_path"], cv2.IMREAD_UNCHANGED)
        anatomy_gt = utils.read_image(dataset_dict["anatomy_path"]).astype("uint8")
        pathology_gt = utils.read_image(dataset_dict["pathology_path"]).astype("uint8")

        utils.check_image_size(dataset_dict, ct)
        utils.check_image_size(dataset_dict, pet)

        
        aug_input = T.AugInput(ct, sem_seg=anatomy_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        
        #Use the same transformation for PET and Pathology
        
        aug_input_pathology = T.AugInput(pet, sem_seg=pathology_gt)
        aug_input_pathology.transform(transforms)

        #Get images and labels 
        ct_aug = aug_input.image
        anatomy_gt_aug = aug_input.sem_seg
        pet_aug = aug_input_pathology.image
        pathology_gt_aug = aug_input_pathology.sem_seg


        #Convert to torch tensors
        ct_aug = torch.as_tensor(ct_aug.astype("float32"))
        pet_aug = torch.as_tensor(pet_aug.astype("float32"))
        anatomy_gt_aug = torch.as_tensor(anatomy_gt_aug.astype("long"))
        pathology_gt_aug = torch.as_tensor(pathology_gt_aug.astype("long"))

        
        stacked_input_img = torch.stack([ct_aug, pet_aug, torch.zeros_like(ct_aug)], dim=0)

        image_shape = (ct_aug.shape[-2], ct_aug.shape[-1])  # h, w

        dataset_dict["image"] = stacked_input_img

        if anatomy_gt_aug is not None:
            dataset_dict["sem_seg"] = anatomy_gt_aug.long()
        if pathology_gt_aug is not None:
            dataset_dict["sem_seg_path"] = pathology_gt_aug.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks for anatomy
        if anatomy_gt_aug is not None:
            anatomy_gt_aug = anatomy_gt_aug.numpy()
            instances = Instances(image_shape)
            classes = np.unique(anatomy_gt_aug)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(anatomy_gt_aug == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, anatomy_gt_aug.shape[-2], anatomy_gt_aug.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
        
        if pathology_gt_aug is not None:
            pathology_gt_aug = pathology_gt_aug.numpy()
            instances_path = Instances(image_shape)
            classes_path = np.unique(pathology_gt_aug)
            # remove ignored region
            classes_path = classes_path[classes_path != self.ignore_label]
            instances_path.gt_classes = torch.tensor(classes_path, dtype=torch.int64)
            
            masks_path = []
            for class_id in classes_path:
                masks_path.append(pathology_gt_aug == class_id)
            if len(masks_path) == 0:
                instances.gt_masks = torch.zeros((0, pathology_gt_aug.shape[-2], pathology_gt_aug.shape[-1]))
            else:
                masks_path = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks_path])
                )
                instances_path.gt_masks = masks_path.tensor
            dataset_dict["instances_path"] = instances_path
        if pathology_gt_aug is None:
            raise Exception("Pathology GT is None")
        return dataset_dict

class CustomM2FInferenceMapper():
    @configurable
    def __init__(self,is_train=False,augmentations=[],
                *,
                ignore_label,
                size_divisibility,
                load_with_numpy,
                ):
        self.is_train = is_train
        self.augmentations = augmentations
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.load_with_numpy = load_with_numpy
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        

    @classmethod
    def from_config(cls, cfg, is_train=False, load_with_numpy=False):
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        ret = {
            "is_train": is_train,
            #"image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "load_with_numpy": load_with_numpy
        }
        return ret
    

    def __call__(self,dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict)
        
        ct = cv2.imread(dataset_dict["file_name"], cv2.IMREAD_UNCHANGED)
        pet = cv2.imread(dataset_dict["pet_path"], cv2.IMREAD_UNCHANGED)

        ct = torch.from_numpy(ct).float()
        pet = torch.from_numpy(pet).float()

        stacked_input_img = torch.stack([ct, pet, torch.zeros_like(ct)], dim=0)

        anatomy_gt = utils.read_image(dataset_dict["anatomy_path"]).astype("uint8")
        pathology_gt = utils.read_image(dataset_dict["pathology_path"]).astype("uint8")



        if anatomy_gt is not None:
            
            assert type(anatomy_gt) is np.ndarray, f"Anatomy GT is not numpy array but {type(anatomy_gt)}"
            instances = Instances(anatomy_gt.shape)
            classes = np.unique(anatomy_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(anatomy_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, anatomy_gt.shape[-2], anatomy_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
        
        if pathology_gt is not None:

            assert type(pathology_gt) is np.ndarray, f"Anatomy GT is not numpy array but {type(pathology_gt)}"
            instances_path = Instances(pathology_gt.shape)
            classes_path = np.unique(pathology_gt)
            # remove ignored region
            classes_path = classes_path[classes_path != self.ignore_label]
            instances_path.gt_classes = torch.tensor(classes_path, dtype=torch.int64)

            masks_path = []
            for class_id in classes_path:
                masks_path.append(pathology_gt == class_id)
            if len(masks_path) == 0:
               instances_path.gt_masks = torch.zeros((0, pathology_gt.shape[-2], pathology_gt.shape[-1]))
            else:
                masks_path = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks_path])
                )
                instances_path.gt_masks = masks_path.tensor
            dataset_dict["instances_path"] = instances_path

        dataset_dict["image"] = stacked_input_img
        return dataset_dict



class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type == "sem_seg":
            evaluator_list.append(
                    SemSegEvaluator(
                        dataset_name,
                        distributed=True,
                        output_dir=output_folder,
                    )
            )
        else:
            raise NotImplementedError(f"No Evaluator for the dataset {dataset_name}")
        
       
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "petctmix":
            mapper = CustomM2FMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "anatomypetctmix":
            mapper = CustomM2FMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg), load_with_numpy=True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "petctmix_separate_anatomy_pathology_labels":
            mapper = CustomM2FMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg), load_with_numpy=False, load_pathology_separately=True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "oct_mapper":
            mapper = CustomM2FMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg), load_with_numpy=False, load_pathology_separately=True, load_with_pil=True, dataype="oct")
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.DATASETS.TEST[0] == "petctmix_dataset_val_full":
            test_loader =  build_detection_test_loader(cfg, dataset_name, mapper=CustomM2FInferenceMapper(cfg,False))
            return test_loader 
        else:    
            raise Exception("Testloader not implemented for the dataset")

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        
        if cfg.MODEL.SEM_SEG_HEAD.WEIGH_CLASSES == "PATHOLOGY_EQUAL":
            patho_weight = cfg.MODEL.SEM_SEG_HEAD.PATHOLOGY_WEIGHT 
            patho_class  = cfg.MODEL.SEM_SEG_HEAD.PATHOLOGY_CLASS 

            if cfg.MODEL.SEM_SEG_HEAD.INVERSE_SCALE:
                model.criterion.empty_weight /= patho_weight
                model.criterion.empty_weight[patho_class] = 1
            else:
                model.criterion.empty_weight[patho_class] = patho_weight

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model


    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue

            #Set the model in the correct inference mode
            
            #Coudld register the same dataset to get anatomy numbers.
            print("Model will be now set into inference mode")
            if isinstance(model,torch.nn.DataParallel) or isinstance(model,torch.nn.parallel.DistributedDataParallel):
                model.module.inference_mode = "pathology"
            else:
                model.inference_mode = "pathology"
                       

            #elif dataset_name == "large_test_val":
            #    model.inference_mode = "pathology"

            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

def setup(args,cfg_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    #Add to be able to merge from weighted loss config file
    add_loss_weights_to_cfg(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_file(args.config_file)
    

    if args.num_gpus == 1:
        cfg.MODEL.RESNETS.NORM = "BN"

    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    prepare_dataset(args)
    
    cfg_merge_files = "model/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml"
    cfg = setup(args,cfg_merge_files)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    print("Trained")

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    #main(args)
