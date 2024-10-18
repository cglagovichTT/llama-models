# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from pathlib import Path
from typing import Optional

import fire

from PIL import Image as PIL_Image
from termcolor import cprint

from models.llama3.api.datatypes import ImageMedia

from models.llama3.reference_impl.generation_tt import Llama


THIS_DIR = Path(__file__).parent.resolve()

import pytest
import os
import ttnn

@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_run_main(
    mesh_device,
    # temperature: float = 0.6,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    mesh_device.enable_program_cache()
    mesh_device.enable_async(True)
    tokenizer_path = str(THIS_DIR.parent / "llama3/api/tokenizer.model")
    ckpt_dir = os.environ["LLAMA_DIR"]
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
        mesh_device=mesh_device,
    )

    with open(THIS_DIR / "resources/dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    with open(THIS_DIR / "resources/pasta.jpeg", "rb") as f:
        img2 = PIL_Image.open(f).convert("RGB")
        
    
    with open(THIS_DIR / "resources/ocr_image.jpeg", "rb") as f:
        ocr_image = PIL_Image.open(f).convert("RGB")
    with open(THIS_DIR / "resources/clutter.jpeg", "rb") as f:
        clutter = PIL_Image.open(f).convert("RGB")

    interleaved_contents = [
        # text only
        # "The color of the sky is blue but sometimes it can also be",
        # image understanding
        # [
        #     ImageMedia(image=img),
        #     "If I had to write a haiku for this one",
        # ],
        [
          ImageMedia(image=ocr_image),
          "The full text in this image is as follows"  
        ],
        # [
        #     ImageMedia(image=clutter),
        #     "The count of vases, books, and miscellaneous items in this image is",  
        # ]
    ]

    for content in interleaved_contents:
        result = generator.text_completion(
            content,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        cprint(f"{content}", end="")
        cprint(f"{result.generation}", color="yellow")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
