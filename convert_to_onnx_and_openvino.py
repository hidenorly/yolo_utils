#!/usr/bin/env python3
# coding: utf-8
#   Copyright 2024 hidenorly
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ultralytics import YOLO
import argparse
import os
import openvino as ov


def inference_result_to_movie(best_weight_path, output_path):
    model = YOLO(best_weight_path)
    model.export(format='onnx')
    pos = best_weight_path.rfind(".")
    if pos!=None:
        ov_model_path = best_weight_path[0:pos]+".onnx"
        ov_model = ov.convert_model(ov_model_path)
        ov.save_model(ov_model, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trained model to openvino model")
    parser.add_argument("-w", "--weight", default='./runs/detect/train/weights/best.pt', help="Path to best weight")
    parser.add_argument("-o", "--output", default='./openvino_model.xml' , help="Output path")
    args = parser.parse_args()
    inference_result_to_movie(args.weight, args.output)
