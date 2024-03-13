# _base_ = "./detic_centernet2_r50_fpn_4x_lvis_boxsup_ishida.py"

# DATA_ROOT_DET = "C:/Users/s.yamagami/Documents/works/classification_tunas/datasets/Phase0dash_80images"
# LOAD_FROM = "work_dirs/detic_centernet2_r50_fpn_4x_lvis_boxsup_ishida--prod/iter_18500.pth"
# ANN_FILE = "annotation/train_annotations.coco.json"
# ANN_FILE_VAL = "annotation/test_annotations.coco.json"
# IMG_PREFIX = dict(img="train/")
# IMG_PREFIX_VAL = dict(img="test/")

# # DATA_ROOT_UNLABEL = "C:/Users/s.yamagami/Documents/works/classification_tunas/datasets/Phase1a_frames"
# # ANN_FILEC_UNLABEL = "annotations/train_annotations.coco.json"
# # IMG_PREFIX_UNLABEL = dict(img="train/")

# DATA_ROOT_CLS = "C:/Users/s.yamagami/Documents/works/classification_tunas/datasets/Yaizu-Phase1/train"
# ANN_FILEC_CLS = "annotations/instances_default.coco.json"
# IMG_PREFIX_CLS = dict(img="images/")

# DATASET_TYPE = "IshidaCocoDataset"

# load_from = LOAD_FROM
# # dataset_type = ['LVISV1Dataset', 'ImageNetLVISV1Dataset']
# dataset_type = [DATASET_TYPE]

# image_size_det = (640, 640)
# image_size_cls = (320, 320)

# # backend = 'pillow'
# backend_args = None

# train_pipeline_det = [
#     dict(type="LoadImageFromFile", backend_args=backend_args),
#     dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
#     dict(type="RandomResize", scale=image_size_det, ratio_range=(0.1, 2.0), keep_ratio=True),
#     dict(type="RandomCrop", crop_type="absolute_range", crop_size=image_size_det, recompute_bbox=True, allow_negative_crop=True),
#     dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type="RandomFlip", prob=0.5),
#     dict(type="PackDetInputs"),
# ]

# train_pipeline_cls = [
#     dict(type="LoadImageFromFile", backend_args=backend_args),
#     dict(type="LoadAnnotations", with_bbox=False, with_label=True),
#     dict(type="RandomResize", scale=image_size_cls, ratio_range=(0.5, 1.5), keep_ratio=True),
#     dict(
#         type="RandomCrop",
#         crop_type="absolute_range",
#         crop_size=image_size_cls,
#         recompute_bbox=False,
#         bbox_clip_border=False,
#         allow_negative_crop=True,
#     ),
#     dict(type="RandomFlip", prob=0.5),
#     dict(type="PackDetInputs"),
# ]

# dataset_det = dict(
#     type="ClassBalancedDataset",
#     oversample_thr=1e-3,
#     dataset=dict(
#         # ------------------------------------------------------
#         type=DATASET_TYPE,
#         data_root=DATA_ROOT_DET,
#         ann_file=ANN_FILE,
#         data_prefix=IMG_PREFIX,
#         # ------------------------------------------------------
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline_det,
#         backend_args=backend_args,
#     ),
# )

# dataset_cls = dict(
#     # ------------------------------------------------------
#     type=DATASET_TYPE,
#     data_root=DATA_ROOT_CLS,
#     ann_file=ANN_FILEC_CLS,
#     data_prefix=IMG_PREFIX_CLS,
#     # ------------------------------------------------------
#     pipeline=train_pipeline_cls,
#     backend_args=backend_args,
# )

# train_dataloader = dict(
#     _delete_=True,
#     batch_size=[8, 32],
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type="MultiDataSampler", dataset_ratio=[1, 4]),
#     batch_sampler=dict(type="MultiDataAspectRatioBatchSampler", num_datasets=2),
#     dataset=dict(type="ConcatDataset", datasets=[dataset_det, dataset_cls]),
# )

# param_scheduler = [
#     dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
#     dict(
#         type="CosineAnnealingLR",
#         begin=0,
#         by_epoch=False,
#         T_max=90000,
#     ),
# ]


# find_unused_parameters = True
