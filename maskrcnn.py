import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from roca.data.cad_manager import register_cads
from roca.data.category_manager import register_categories


metadata={'scenes': os.path.abspath('/home/aston/Desktop/python/ROCA/metadata/scannetv2_train.txt')}
name = 'scan2cad'
json_file = '/home/aston/Desktop/python/ROCA/Data/Dataset/scan2cad_instances_train.json'
data_root = '/home/aston/Desktop/python/ROCA/Data'
image_root = os.path.join(data_root, 'Images')
rendering_root = os.path.join(data_root, 'Rendering')
full_annot = os.path.join(data_root, 'full_annotations.json')
extra_keys = ['t', 'q', 's', 'intrinsics', 'alignment_id', 'model', 'id']
DatasetCatalog.register(
    name, lambda: load_coco_json(json_file, image_root, name, extra_keys)
)

# Fill lazy loading stuff
DatasetCatalog.get(name)

MetadataCatalog.get(name).set(
    json_file=json_file,
    image_root=image_root,
    evaluator_type='coco',
    rendering_root=rendering_root,
    full_annot=full_annot,
    **metadata
)

test_meta = MetadataCatalog.get(name)
data_dict = DatasetCatalog.get(name)
import random
from detectron2.utils.visualizer import Visualizer
import cv2
for d in random.sample(data_dict,3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=test_meta, scale=0.5)  # cv2 RGB 需要倒序
    vis = visualizer.draw_dataset_dict(d)  # 字典
    cv2.imshow("",vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

