import os
import shutil
# copying a file into transformers because of a suspected bug
# shutil.copy(os.path.join(os.path.dirname(__file__), "./model/configuration_utils.py"), "/usr/local/lib/python3.12/dist-packages/transformers/configuration_utils.py")

import albumentations
import torch
import yaml
import numpy as np
import gradio as gr
from pathlib import Path
from functools import partial
from terratorch.datamodules import Sen1Floods11NonGeoDataModule
from inference import process_channel_group, _convert_np_uint8, load_example, run_model
from prithvi_mae import PrithviMAE

import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

token = os.environ.get("HF_TOKEN", None)

def extract_rgb_imgs(input_img, pred_img, channels):
    """ Wrapper function to save Geotiff images (original, reconstructed, masked) per timestamp.
    Args:
        input_img: input torch.Tensor with shape (C, H, W).
        rec_img: reconstructed torch.Tensor with shape (C, T, H, W).
        pred_img: mask torch.Tensor with shape (C, T, H, W).
        channels: list of indices representing RGB channels.
        mean: list of mean values for each band.
        std: list of std values for each band.
        output_dir: directory where to save outputs.
        meta_data: list of dicts with geotiff meta info.
    """
    rgb_orig_list = []
    rgb_mask_list = []
    rgb_pred_list = []

    for t in range(input_img.shape[1]):
        rgb_orig, rgb_pred = process_channel_group(orig_img=input_img[:, t, :, :],
                                                   new_img=rec_img[:, t, :, :],
                                                   channels=channels,
                                                   mean=mean,
                                                   std=std)

        rgb_mask = mask_img[channels, t, :, :] * rgb_orig

        # extract images
        rgb_orig_list.append(_convert_np_uint8(rgb_orig).transpose(1, 2, 0))
        rgb_mask_list.append(_convert_np_uint8(rgb_mask).transpose(1, 2, 0))
        rgb_pred_list.append(_convert_np_uint8(rgb_pred).transpose(1, 2, 0))

    # Add white dummy image values for missing timestamps
    dummy = np.ones((20, 20), dtype=np.uint8) * 255
    num_dummies = 4 - len(rgb_orig_list)
    if num_dummies:
        rgb_orig_list.extend([dummy] * num_dummies)
        rgb_mask_list.extend([dummy] * num_dummies)
        rgb_pred_list.extend([dummy] * num_dummies)

    outputs = rgb_orig_list + rgb_mask_list + rgb_pred_list

    return outputs


def predict_on_images(data_file: str | Path, config_path: str, model: PrithviMAE):
    try:
        data_file = data_file.name
        print('Path extracted from example')
    except:
        print('Files submitted through UI')

    # Get parameters --------
    print('This is the printout', data_file)

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    img_size = 256  # Size of Sen1Floods11

    # Loading data ---------------------------------------------------------------------------------

    input_data, temporal_coords, location_coords, meta_data = load_example(file_paths=[data_file])

    if input_data.shape[1] == 6:
        pass
    elif input_data.shape[1] == 13:
        input_data = input_data[:, [1,2,3,8,11,12], ...]
    else:
        raise Exception(f'Input data has {input_data.shape[1]} channels. Expect either 6 Prithvi channels or 13 S2L1C channels.')

    if input_data.mean() > 1:
        input_data = input_data / 10000  # Convert to range 0-1

    # Running model --------------------------------------------------------------------------------


    channels = [config_dict['data']['init_args']['bands'].index(b) for b in ["RED", "GREEN", "BLUE"]]  # BGR -> RGB

    datamodule_config = {
                    'bands': ['BLUE',
                         'GREEN',
                         'RED',
                         'NIR_NARROW',
                         'SWIR_1',
                         'SWIR_2'],
                   'batch_size': 16,
                   'constant_scale': 0.0001,
                   'data_root': '/dccstor/geofm-finetuning/datasets/sen1floods11',
                   'drop_last': True,
                   'no_data_replace': 0.0,
                   'no_label_replace': -1,
                   'num_workers': 8,
                   'test_transform': [albumentations.Resize(always_apply=False,
                                                     height=448,
                                                     interpolation=1,
                                                     p=1,
                                                     width=448),
                                      albumentations.pytorch.ToTensorV2(
                                          transpose_mask=False,
                                          always_apply=True,
                                                     p=1.0
                                                     )],
    }

    datamodule = Sen1Floods11NonGeoDataModule(data_root=datamodule_config['data_root'],
                                              batch_size=datamodule_config["batch_size"],
                                              num_workers=datamodule_config["num_workers"],
                                              bands=datamodule_config["bands"],
                                              drop_last=datamodule_config["drop_last"],
                                              test_transform=datamodule_config["test_transform"
                                                                               ""])
    pred = run_model(input_data, temporal_coords, location_coords,
                     model, datamodule, img_size)

    if input_data.mean() < 1:
        input_data = input_data * 10000  # Scale to 0-10000

    # Extract RGB images for display
    rgb_orig = process_channel_group(
        orig_img=torch.Tensor(input_data[0, :, 0, ...]),
        channels=channels,
    )
    out_rgb_orig = _convert_np_uint8(rgb_orig).transpose(1, 2, 0)
    out_pred_rgb = _convert_np_uint8(pred).repeat(3, axis=0).transpose(1, 2, 0)

    pred[pred == 0.] = np.nan
    img_pred = rgb_orig * 0.6 + pred * 0.4
    img_pred[img_pred.isnan()] = rgb_orig[img_pred.isnan()]

    out_img_pred = _convert_np_uint8(img_pred).transpose(1, 2, 0)

    outputs = [out_rgb_orig] + [out_pred_rgb] + [out_img_pred]

    print("Done!")

    return outputs

print("################ Loading model to vLLM ##############")
model_obj = PrithviMAE()
print("################ Model loading done ##############")
config_path = os.path.join(os.path.dirname(__file__), "./model/config.yaml")
run_inference = partial(predict_on_images, model=model_obj, config_path=config_path)

with gr.Blocks() as demo:
    gr.Markdown(value='# Prithvi-EO-2.0 Sen1Floods11 Demo')
    gr.Markdown(value='''
Prithvi-EO-2.0 is the second generation EO foundation model developed by the IBM and NASA team.
This demo showcases the fine-tuned Prithvi-EO-2.0-300M-TL model to detect water using Sentinel 2 imagery from on the [sen1floods11 dataset](https://github.com/cloudtostreet/Sen1Floods11). More details can be found [here](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11).\n

The user needs to provide a Sentinel-2 L1C image with either all the 13 bands or the six Prithvi bands (Blue, Green, Red, Narrow NIR, SWIR, SWIR 2). The demo code selects the required bands.
We recommend submitting images of 500 to ~1000 pixels for faster processing time. Images bigger than 256x256 are processed using a sliding window approach which can lead to artefacts between patches.\n
Optionally, the location information is extracted from the tif files while the temporal information can be provided in the filename in the format `<date>T<time>` or `<year><julian day>T<time>` (HLS format). 
Some example images are provided at the end of this page.
''')
    with gr.Row():
        with gr.Column():
            inp_file = gr.File(elem_id='file')
            # inp_slider = gr.Slider(0, 100, value=50, label="Mask ratio", info="Choose ratio of masking between 0 and 100", elem_id='slider'),
            btn = gr.Button("Submit")
    with gr.Row():
        gr.Markdown(value='## Input image')
        gr.Markdown(value='## Prediction*')
        gr.Markdown(value='## Overlay')

    with gr.Row():
        original = gr.Image(image_mode='RGB', show_label=False, show_fullscreen_button=False)
        predicted = gr.Image(image_mode='RGB', show_label=False, show_fullscreen_button=False)
        overlay = gr.Image(image_mode='RGB', show_label=False, show_fullscreen_button=False)

    gr.Markdown(value='\* White = flood; Black = no flood')

    btn.click(fn=run_inference,
              inputs=inp_file,
              outputs=[original] + [predicted] + [overlay])

    with gr.Row():
        gr.Examples(examples=[
            os.path.join(os.path.dirname(__file__), "examples/India_900498_S2Hand.tif"),
            os.path.join(os.path.dirname(__file__), "examples/Spain_7370579_S2Hand.tif"),
            os.path.join(os.path.dirname(__file__), "examples/USA_430764_S2Hand.tif")],
            inputs=inp_file,
                    outputs=[original] + [predicted] + [overlay],
                    fn=run_inference,
                    cache_examples=False
    )

demo.launch(share=False, ssr_mode=False, server_name="0.0.0.0")
