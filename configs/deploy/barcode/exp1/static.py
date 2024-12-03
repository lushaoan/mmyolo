"""
Author: Lu ShaoAn
Brief: 
Version: 0.1
Date: 2024-10-30 11:55:31
Copyright: Copyright (c) 2022
LastEditTime: 2024-10-30 11:55:31
"""

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="end2end.onnx",
    input_names=["input"],
    output_names=["labels_0", "labels_1", "labels_2", "dets_0", "dets_1", "dets_2"],
    # output_names=["dets", "labels"],
    input_shape=None,
    optimize=True,
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
