#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
from pathlib import Path
import tqdm
from PIL import Image


def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output)


## 这里要先link上: 创建一个符号链接 (symlink) 从 $Painter_ROOT/datasets/ade20k 到 $Painter_ROOT/datasets/ADEChallengeData2016
# 遍历"ADEChallengeData2016"目录下的"annotations"子目录中的"training"和"validation"子目录
## 并将这些目录中的文件转换为位于"annotations_detectron2"子目录下相应子目录中的输出文件

if __name__ == "__main__":
    # 创建一个指向名为"ADEChallengeData2016"的目录的Path对象
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "ADEChallengeData2016" 
    print(dataset_dir)
    # dataset_dir = '/hhd3/ld/data/ADEChallengeData2016'
    for name in ["training", "validation"]:
        # 创建一个新的Path对象，指向"annotations_detectron2"子目录下的name目录 (name将是"training"或"validation")
        annotation_dir = dataset_dir / "ade20k" / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            convert(file, output_file)
