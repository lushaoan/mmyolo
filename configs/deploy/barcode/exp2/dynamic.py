"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-11-22 09:59:56
Copyright: Copyright (c) 2024
LastEditTime: 2024-11-22 09:59:57
"""

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="yolo_input_dynamic.onnx",
    input_names=["input"],
    # output_names=["dets", "labels"],
    output_names=["labels_0", "labels_1", "labels_2", "dets_0", "dets_1", "dets_2"],
    input_shape=None,
    optimize=True,
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        # "labels_0": {0: "batch", 1: "num_dets"},
        # "labels_1": {0: "batch", 1: "num_dets"},
        # "labels_2": {0: "batch", 1: "num_dets"},
        # "dets_0": {0: "batch", 1: "num_dets"},
        # "dets_1": {0: "batch", 1: "num_dets"},
        # "dets_2": {0: "batch", 1: "num_dets"},
    },
)

codebase_config = dict(
    type="mmyolo",
    task="ObjectDetection",
    model_type="end2end",
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ),
    module=["mmyolo.deploy"],
)
backend_config = dict(type="onnxruntime")
