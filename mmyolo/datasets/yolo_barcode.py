"""
Author: Lu ShaoAn, Smartmore Corporation
Brief: 
Version: 0.1
Date: 2024-10-14 18:29:07
Copyright: Copyright (c) 2022
"""

import json
import os
from typing import List, Sequence

import cv2
import numpy as np
from mmdet.datasets import BaseDetDataset

from ..registry import DATASETS


@DATASETS.register_module()
class YoloBarcodeDataset(BaseDetDataset):
    def __init__(self, *args, train_file_infos: List[dict], **kwargs):
        self.train_file_infos = train_file_infos
        self.label_map = {"bar": 0, "qr": 1, "dm": 2, "pdf417": 3}

        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        for file_info in self.train_file_infos:
            root_path = file_info["root"]
            ann_file_path = file_info["path"]
            with open(ann_file_path, "r") as train_file:
                lines = train_file.readlines()
                for line in lines:
                    data_info = {}

                    img_path = os.path.join(root_path, line).split("\n")[0]
                    file_name, _ = os.path.splitext(img_path)
                    json_path = file_name + ".json"
                    with open(json_path, "r") as json_file:
                        json_data = json.load(json_file)

                        data_info["img_path"] = img_path
                        data_info["height"] = json_data["imageHeight"]
                        data_info["width"] = json_data["imageWidth"]

                        instances = []
                        for shape in json_data["shapes"]:
                            instance = {}
                            bbox_pts = np.array(shape["points"], dtype=np.float32)

                            if len(bbox_pts) != 4:
                                continue

                            for i in range(0, 4):
                                bbox_pts[i][0] = np.clip(
                                    bbox_pts[i][0], 0, data_info["width"] - 1
                                )
                                bbox_pts[i][1] = np.clip(
                                    bbox_pts[i][1], 0, data_info["height"] - 1
                                )

                            if shape["label"].lower() not in self.label_map:
                                continue

                            x1, y1, w, h = cv2.boundingRect(bbox_pts)
                            bbox = [x1, y1, x1 + w, y1 + h]
                            instance["bbox"] = bbox
                            instance["bbox_label"] = self.label_map[
                                shape["label"].lower()
                            ]
                            instance["ignore_flag"] = 0

                            instances.append(instance)

                        data_info["instances"] = instances
                        data_list.append(data_info)

        return data_list
