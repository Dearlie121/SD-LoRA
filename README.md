# LoRA for SD fine-tuning.

> Using LoRA to fine tune on Hanok dataset

## Modified site-packages/lora_diffusion/dataset.py

Images had their captions in a separate caption.txt files. So the above code was changed to below.

```bash
self.captions = [
    x.split("/")[-1].split(".")[0] for x in self.instance_images_path
]
```
```bash
self.captions = open(f"{instance_data_root}/caption.txt").readlines()
```

## Train.sh

To define the directories and execute lora_pti.py with hyperparameters, used a shell script.
INSTANCE_DIR, OUTPUT_DIR, --use_template <-- were modified

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="../data/modern/ext"
export OUTPUT_DIR="../exps/output_ext2"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --scale_lr \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens="<s1>|<s2>" \
  --use_template=None\
#  --use_template="style"\
  --save_steps=100 \
  --max_train_steps_ti=1000 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=1 \
```

## Img2img inference

For ineference.
1 Base SD model
2 apply LoRA
3 turn off unet & text_encoder
4 scale unet & text_encoder to 0.7

```python
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from lora_diffusion import patch_pipe, tune_lora_scale

path = './data/hamyang/'
file_list = os.listdir(path)

for i in file_list:
    if i.endswith('.png'):
        torch.manual_seed(42)
        model_id = "runwayml/stable-diffusion-v1-5"
        prompt = "a contemporary house for camping near a river or a lake"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        name = i.split('.')[0]
        init_image = Image.open(path+i).convert("RGB").resize((512, 512))
        image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
        image = image.save(path+'output5/'+f'{name}1.png')
        
        patch_pipe(pipe, "./exps/output_ext2/final_lora.safetensors", patch_text=True, patch_unet=True, patch_ti=True)
        image1 = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
        image1 = image1.save(path+'output5/'+f'{name}2.png')
        
        tune_lora_scale(pipe.unet, 0.0)
        tune_lora_scale(pipe.text_encoder, 0.0)
        image2 = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
        image2 = image2.save(path+'output5/'+f'{name}3.png')

        tune_lora_scale(pipe.unet, 0.7)
        tune_lora_scale(pipe.text_encoder, 0.3)
        image3 = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
        image3 = image3.save(path+'output5/'+f'{name}4.png')
    else:
        continue
```
