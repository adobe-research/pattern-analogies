---
license: other
license_name: adobe-research-license
license_link: LICENSE
library_name: diffusers
base_model:
- shi-labs/versatile-diffusion
pipeline_tag: image-to-image
tags:
- Editing
- Analogy
- Patterns
---


# Pattern Analogies V1.0 Model Card

This respository contains TriFuser --- a diffusion model trained for analogical editing of pattern images as a part of our recent tech report **"Pattern Analogies: Learning to Perform Programmatic Image Edits by Analogy"**.


## Abstract

Pattern images are everywhere in the digital and physical worlds, and tools to edit them are valuable. But editing pattern images is tricky: desired edits are often \emph{programmatic}: structure-aware edits that alter the underlying program which generates the pattern. One could attempt to infer this underlying program, but current methods for doing so struggle with complex images and produce unorganized programs that make editing tedious. In this work, we introduce a novel approach to perform programmatic edits on pattern images. By using a \emph{pattern analogy}---a pair of simple patterns to demonstrate the intended edit---and a learning-based generative model to execute these edits, our method allows users to intuitively edit patterns. To enable this paradigm, we introduce SplitWeaver, a domain-specific language that, combined with a framework for sampling synthetic pattern analogies, enables the creation of a large, high-quality synthetic training dataset.We also present TriFuser, a Latent Diffusion Model (LDM) designed to overcome critical issues that arise when naively deploying LDMs to this task. Extensive experiments on real-world, artist-sourced patterns reveals that our method faithfully performs the demonstrated edit while also generalizing to related pattern styles beyond its training distribution.

Please check out our [preprint]() for more information.

## Model Details

TriFuser model uses the image-variation model of [Versatile Diffusion](https://huggingface.co/shi-labs/versatile-diffusion) as the starting point. It takes three images as input, (A, A*, B), and generates image B* as output, which satisfies the analogical relation A:A*::B:B*. The figure below shows the architecture of TriFuser in detail. Please read our pre-print for more information.



One single flow of Versatile Diffusion contains a VAE, a diffuser, and a context encoder,  and thus handles one task (e.g., text-to-image) under one data type (e.g., image) and one context type (e.g., text). The multi-flow structure of Versatile Diffusion shows in the following diagram:

<p align="center">
  <img src="https://huggingface.co/bardofcodes/pattern_analogies/resolve/main/assets/arch_1.png" width="99%">
</p>
<p align="center">
  <img src="https://huggingface.co/bardofcodes/pattern_analogies/resolve/main/assets/arch_2.png" width="99%">
</p>

- **Developed by:** Aditya Ganeshan, Thibault Groueix, Paul Guerrero, Radomír Měch, Matthew Fisher and Daniel Ritchie
- **Model type:** Diffusion-based image2image generative model
- **Language(s):** English
- **License:** Adobe Research License
- **Resources for more information:** More information along with training code will be released in this [GitHub Repository](https://github.com/bardofcodes/pattern_analogies).

## Citation

TBD

## Usage

You can use the model with the [🧨Diffusers library](https://github.com/huggingface/diffusers).


### PatternAnalogiesTrifuser

This repository contains example inputs to demonstrate the model's capabilities. Please change `EXAMPLE_ID` from 0-9 to check out the different examples. 

```py
import requests
import torch as th
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from diffusers import DiffusionPipeline

SEED = 1729
DEVICE = th.device("cuda")
DTYPE = th.float16
FIG_K = 3
EXAMPLE_ID = 0

# Now we need to do the trick
pretrained_path = "bardofcodes/pattern_analogies"
new_pipe = DiffusionPipeline.from_pretrained(
    pretrained_path, 
    custom_pipeline=pretrained_path, 
    trust_remote_code=True
)

img_urls = [
    f"https://huggingface.co/bardofcodes/pattern_analogies/resolve/main/examples/{EXAMPLE_ID}_a.png",
    f"https://huggingface.co/bardofcodes/pattern_analogies/resolve/main/examples/{EXAMPLE_ID}_a_star.png",
    f"https://huggingface.co/bardofcodes/pattern_analogies/resolve/main/examples/{EXAMPLE_ID}_b.png",
]
images = []
for url in img_urls:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)

pipe_input = [tuple(images)]

pipe = new_pipe.to(DEVICE, DTYPE)
var_images = pipe(pipe_input, num_inference_steps=50, num_images_per_prompt=3,).images

plt.figure(figsize=(3*FIG_K, 2*FIG_K))
plt.axis('off')
plt.legend(framealpha=1)
plt.rcParams['legend.fontsize'] = 'large'
for i in range(6):
    if i == 0:
        plt.subplot(2, 3, i+1)
        val_image = img1
        label_str = "A"
    elif i == 1:
        plt.subplot(2, 3, i+1)
        val_image = alt_img1
        label_str = "A*"
    elif i == 2:
        plt.subplot(2, 3, i+1)
        val_image = img2
        label_str = "Target"
    else:
        plt.subplot(2, 3,i + 1)
        val_image = var_images[i-3]
        label_str = f"Variation {i-2}"
    
    val_image = ImageOps.expand(val_image,border=2,fill='black')
    plt.imshow(val_image)
    plt.scatter([], [], c="r", label=label_str)
    plt.legend(loc="lower right")
    plt.axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.01)
```

### Full GitHub Repository

Will be released soon.

## Cautions, Biases, and Content Acknowledgment

We would like the raise the awareness of users of this demo of its potential issues and concerns. Like previous large foundation models, the use of our model could be problematic in some cases, partially due to the imperfect training data and pretrained network (VAEs / context encoders) with limited scope. We welcome researchers and users to report issues with the HuggingFace community discussion feature or email the authors.

However, since our model targets the task of editing images, it is strongly guided by the user input. To the best of our knowledge, with sanitized inputs, our model outputs sanitized outputs consistently.