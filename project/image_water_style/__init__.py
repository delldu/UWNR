"""Image/Video Photo Style Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

import todos
from . import water_style

import pdb


def get_water_style_model():
    """Create model."""

    model = water_style.ImageWaterStyleModel()
    model = todos.model.ResizePadModel(model)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_water_style.torch"):
        model.save("output/image_water_style.torch")

    return model, device


def image_predict(input_files, style_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_water_style_model()

    # load files
    image_filenames = todos.data.load_files(input_files)
    style_filenames = todos.data.load_files(style_files)

    # pdb.set_trace()

    assert len(image_filenames) == len(style_filenames), "Number of style files should match content files"


    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for c_filename, s_filename in zip(image_filenames, style_filenames):
        progress_bar.update(1)

        # orig input
        content_tensor = todos.data.load_tensor(c_filename)
        B, C, H, W = content_tensor.shape

        style_tensor = todos.data.load_tensor(s_filename)
        style_tensor = F.interpolate(style_tensor, (H, W), mode="bilinear", align_corners=True)

        # input_tensor =  content_style + style_tensor
        input_tensor = torch.cat((content_tensor, style_tensor), dim=1)
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(c_filename)}"

        todos.data.save_tensor([content_tensor, style_tensor, predict_tensor], output_file)

    todos.model.reset_device()
