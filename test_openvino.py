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

import cv2
from ultralytics import YOLO
import argparse
import os

def inference_result_to_movie(model_path, video_path, result_path):
    model = YOLO(model_path)
    if video_path:
        cap = cv2.VideoCapture(video_path)
        video_writer = None

        annotated_frames = []
        while cap.isOpened():
            ret, frame = cap.read()

            annotated_frame = None
            if ret:
                results = model.track(frame, persist=True) # 物体をトラッキング
                annotated_frame = results[0].plot()
                # enqueue annoted_frames and initialize video_writer
                annotated_frames.append(annotated_frame)
                if result_path and not video_writer:
                    height, width, layers = annotated_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(result_path, fourcc, 30, (width, height))
            else:
                break

            # write annoted_frames periodically
            if video_writer and len(annotated_frames) > 60:
                for frame in annotated_frames:
                    video_writer.write(frame)
                annotated_frames.clear()

        # finalizing...
        if video_writer:
            for frame in annotated_frames:
                video_writer.write(frame)
            annotated_frames.clear()
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset for YOLO training")
    parser.add_argument("-i", "--input", required=True, help="Input video path. file or rtsp:, etc.")
    parser.add_argument("-o", "--output", default='./annotated_video.mp4' , help="Output path")
    parser.add_argument("-m", "--model", default='./openvino_model.xml', help="Path to model")
    args = parser.parse_args()

    inference_result_to_movie(args.model, args.input, args.output)
