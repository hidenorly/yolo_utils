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

import os
import shutil
import random
import yaml
import argparse

def ensure_output_folder(output_dir):
    train_image_folder = os.path.join(output_dir, "datasets", "train", "images")
    train_label_folder = os.path.join(output_dir, "datasets", "train", "labels")
    valid_image_folder = os.path.join(output_dir, "datasets", "valid", "images")
    valid_label_folder = os.path.join(output_dir, "datasets", "valid", "labels")

    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(valid_image_folder, exist_ok=True)
    os.makedirs(valid_label_folder, exist_ok=True)

    return train_image_folder, train_label_folder, valid_image_folder, valid_label_folder


def get_files_per_label(label_dir, label_files, image_files):
    data_per_label_id = {}
    i = 0
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        if os.path.isfile(label_path):
            lines = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                _id = line.split(' ')[0]
                if not _id in data_per_label_id:
                    data_per_label_id[_id] = []
                data_per_label_id[_id].append( {"id":_id, "label_file":label_file, "image_file":image_files[i], "index": i} )
            i = i + 1
    return data_per_label_id


def copy_data_to_dest(image_dir, image_file, image_target, label_dir, label_file, label_target):
    image_file = os.path.join(image_dir, image_file)
    label_file = os.path.join(label_dir, label_file)
    if os.path.isfile(image_file) and os.path.isfile(label_file):
        shutil.copy(image_file, image_target)
        shutil.copy(label_file, label_target)


def split_dataset(image_dir, label_dir, output_dir, classes_path, train_ratio):
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    train_image_folder, train_label_folder, valid_image_folder, valid_label_folder = ensure_output_folder(output_dir)
    data_per_label_id = get_files_per_label(label_dir, label_files, image_files)

    class_names=[]
    with open(classes_path, 'r') as f:
        for line in f:
            class_names.append( line.strip() )
    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for class_name in class_names:
            f.write(f'{class_name}\n')

    for _id, label_data in data_per_label_id.items():
        num_train = int(len(label_data) * train_ratio)
        random.shuffle(label_data)
        train_data = label_data[:num_train]
        valid_data = label_data[num_train:]

        for _data in train_data:
            copy_data_to_dest(image_dir, _data["image_file"], train_image_folder, label_dir, _data["label_file"], train_label_folder)
        for _data in valid_data:
            copy_data_to_dest(image_dir, _data["image_file"], valid_image_folder, label_dir, _data["label_file"], valid_label_folder)

    # generate yaml
    data = {
        'train': "./train/images/",
        'val': "./valid/images/",
        'nc': len(data_per_label_id),
        'names': class_names
    }
    with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset for YOLO training")
    parser.add_argument("-i", "--image_dir", required=True, help="Directory containing images")
    parser.add_argument("-l", "--label_dir", required=True, help="Directory containing labels")
    parser.add_argument("-c", "--classes", required=True, help="Path for classses.txt")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    parser.add_argument("-r", "--train_ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    args = parser.parse_args()

    split_dataset(args.image_dir, args.label_dir, args.output_dir, args.classes, args.train_ratio)
