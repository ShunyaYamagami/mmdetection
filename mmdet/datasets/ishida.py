from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

# from mmdet.datasets.builder import DATASETS
# from mmdet_extension.datasets.semi_dataset import SemiDataset
# from mmdet_extension.datasets.txt_style import TXTDataset
from pycocotools.coco import COCO

ISHIDA_CLASSES = ("Skipjack tuna", "Thunnus", "Unknown")


# @DATASETS.register_module()
# class IshidaSemiDataset(SemiDataset):
#     CLASSES = ISHIDA_CLASSES
#     # def __init__(self, *args, **kwargs):
#     #     super().__init__(*args, **kwargs)

#     def get_data_cls(self, ann_file):
#         if ann_file.endswith(".json"):
#             return IshidaCocoDataset
#         elif ann_file.endswith(".txt"):
#             return IshidaTXTDataset
#         else:
#             raise ValueError(f"please use json or text format annotations")


@DATASETS.register_module()
class IshidaCocoDataset(CocoDataset):
    # mmdetのversionによって、CLASSESを使う場合とMETAINFOを使う場合があるらしい
    CLASSES = ISHIDA_CLASSES

    METAINFO = {
        'classes':
        ISHIDA_CLASSES,
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
    }
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    @classmethod
    def get_classes(cls, classes=None):
        return ISHIDA_CLASSES

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        # mmdetのversionによってここで設定する必要がある
        # self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info["filename"] = info["file_name"]
            data_infos.append(info)
        return data_infos


# @DATASETS.register_module()
# class IshidaTXTDataset(TXTDataset):
#     CLASSES = ISHIDA_CLASSES
#     # def __init__(self, *args, **kwargs):
#     #     super().__init__(*args, **kwargs)

#     @classmethod
#     def get_classes(cls, classes=None):
#         return ISHIDA_CLASSES
