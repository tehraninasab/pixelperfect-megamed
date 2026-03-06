# Pixel Perfect MegaMed
**A Megapixel-Scale Vision–Language Foundation Model for High-Resolution Medical Image Generation**

<p align="center">
  <a href="https://tehraninasab.github.io/pixelperfect-megamed/">🌐 Project Page</a> •
  <a href="https://www.arxiv.org/abs/2507.12698">📄 arXiv</a> •
  <a href="https://link.springer.com/chapter/10.1007/978-3-032-05472-2_27">📘 Springer</a> •
  <a href="https://huggingface.co/tehraninasab/pixelperfect-megamed-lora">🤗 Hugging Face</a>
</p>


---

## 📌 Overview

**Pixel Perfect MegaMed** is a **megapixel-scale vision–language foundation model** for **high-resolution medical image generation**, designed to overcome the resolution bottlenecks of existing diffusion-based medical imaging models.

Built on **Stable Diffusion XL (SDXL)** and the latest **parameter-efficient fine-tuning (PEFT)** techniques, Pixel Perfect MegaMed enables **progressive synthesis of medical images up to megapixel resolution** while maintaining semantic fidelity and anatomical realism.

This repository contains the **training code, evaluation pipelines, and utilities** used in the paper.

## 🤗 Model Weights

The trained LoRA weights for Pixel Perfect MegaMed are available on Hugging Face:

👉 https://huggingface.co/tehraninasab/pixelperfect-megamed-lora

These weights are designed to be used with the Stable Diffusion XL base model.

# 🚀 Usage
**1. Install dependencies**
```
pip install diffusers transformers accelerate torch
```

**2. Load the model with LoRA weights**
```
import torch
from diffusers import StableDiffusionXLPipeline

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.load_lora_weights("tehraninasab/pixelperfect-megamed-lora")

prompt = "Chest X-ray showing pleural effusion"

image = pipe(
    prompt=prompt,
    num_inference_steps=50
).images[0]

image.save("sample.png")
```

**3. Ultra-high resolution generation**

Pixel Perfect MegaMed supports ultra-high-resolution synthesis (up to 2048×2048) using DemoFusion.

Example usage:
```
import torch
from demofusion.pipeline_demofusion_sdxl import DemoFusionSDXLPipeline

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_repo = "tehraninasab/pixelperfect-megamed-lora"

pipe = DemoFusionSDXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights(lora_repo)

prompt = "Chest X-ray showing pneumothorax"

image = pipe(
    prompt=prompt,
    height=2048,
    width=2048,
    num_inference_steps=50
).images[0]

image.save("megamed_demofusion.png")
```

# 📚 Citation

If you use **Pixel Perfect MegaMed** in your research, please cite:

```bibtex
@inproceedings{tehraninasab2025pixel,
  title     = {Pixel Perfect MegaMed: A Megapixel-Scale Vision-Language Foundation Model for Generating High Resolution Medical Images},
  author    = {TehraniNasab, Zahra and Ni, Hujun and Kumar, Amar and Arbel, Tal},
  booktitle = {MICCAI Workshop on Deep Generative Models},
  pages     = {277--287},
  year      = {2025},
  publisher = {Springer}
}
```

# 📜 License

The **Pixel Perfect MegaMed LoRA weights** are released under the  
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

See the LICENSE file for details.

Because this work builds upon **Stable Diffusion XL**, users must also comply with the Stable Diffusion XL license (CreativeML OpenRAIL-M).

# ⚠️ Disclaimer
Pixel Perfect MegaMed is a research prototype intended for academic and research purposes only. 

The generated images are synthetic and should not be used for clinical diagnosis, medical decision-making, or patient care.

This project is built upon publicly available datasets including CheXpert and MIMIC-CXR, and uses the Stable Diffusion XL architecture. Users must comply with the respective dataset and model licenses when using this work.

The authors and contributors make no guarantees regarding the medical validity, safety, or clinical applicability of the generated images, and assume no responsibility for any use outside of research contexts.
