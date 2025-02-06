from typing import Tuple
from einops import rearrange
import os
from timm.layers import to_2tuple
from vllm import LLM
import torch
import time
class PrithviMAE:
    def __init__(self):
        print("Initializing PrithviMAE model")
        self.model = LLM(model=os.path.join(os.path.dirname(__file__), "./model"), skip_tokenizer_init=True, dtype="float32")

    def get_patch_size(self) -> Tuple[int, int, int]:
        return self.model.llm_engine.model_config.hf_config.to_dict()["pretrained_cfg"]["patch_size"]

    def get_img_size(self) -> Tuple[int, int]:
        img_size = self.model.llm_engine.model_config.hf_config.to_dict()["pretrained_cfg"]["img_size"]
        return to_2tuple(img_size)

    def get_num_channels(self) -> int:
        return self.model.llm_engine.model_config.hf_config.to_dict()["pretrained_cfg"]["in_chans"]

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (torch.FloatTensor of shape `(batch_size, num_channels, time, height, width)`):
                Pixel values.

        Returns:
            torch.FloatTensor of shape `(batch_size, num_patches, patch_size[0]*patch_size[1]*patch_size[2] * num_channels)`:
                Patchified pixel values.
        """
        patch_size_t, patch_size_h, patch_size_w = self.get_patch_size()
        num_channels = self.get_num_channels()

        # patchify
        patchified_pixel_values = rearrange(pixel_values, 'b c (t s) (h p) (w q) -> b (t h w) (s p q c)',
                                            c=num_channels, s=patch_size_t, p=patch_size_h, q=patch_size_w)


        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values, image_size: Tuple[int, int] | None = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape
                `(batch_size, num_patches, patch_size[0]*patch_size[1]*patch_size[2] * num_channels)`:
                Patchified pixel values.
            image_size (`Tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size_t, patch_size_h, patch_size_w = self.get_patch_size()
        image_size = to_2tuple(image_size) if image_size is not None else self.get_img_size()
        original_height, original_width = image_size
        num_patches_h = original_height // patch_size_h
        num_patches_w = original_width // patch_size_w
        num_channels = self.get_num_channels()

        pixel_values = rearrange(patchified_pixel_values, 'b (t h w) (s p q c) -> b c (t s) (h p) (w q)',
                                 c=num_channels, h=num_patches_h, w=num_patches_w,
                                 s=patch_size_t, p=patch_size_h, q=patch_size_w)
        return pixel_values

    def run(self, input_data, temporal_coords, location_coords):
        print("################ Running inference on vLLM ##############")
        # merge the inputs into one data structure
        mm_data = {
            "pixel_values": torch.empty(0) if input_data is None else input_data,
            "location_coords": torch.empty(0) if location_coords is None else location_coords
        }

        prompt = {
            "prompt_token_ids": [1],
            "multi_modal_data": mm_data
        }

        start = time.time()
        outputs = self.model.encode(prompt, use_tqdm=False)
        end = time.time()
        elapsed = end - start
        print(f"################ Inference done (it took {round(elapsed,2)} seconds)  ##############")

        return outputs[0].outputs.data