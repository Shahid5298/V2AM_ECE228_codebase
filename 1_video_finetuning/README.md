# Stage 1 — Video-model fine-tuning (image-to-video LoRA)

LoRA-fine-tunes an image-to-video (i2v) diffusion model on MimicGen
demonstrations. Given the **current camera frame** and a **task prompt**, the
fine-tuned model generates the next few frames of the task. These "imagined"
future frames are later used as a conditioning stream for the action head
(stage 2).

The pretrained i2v **base-model weights** are large and are *not* included in
this repo. Point the `HUMMINGBIRD_I2V` environment variable at the i2v model
tree, and place the base checkpoints under its `hum_infer/checkpoints/`.

## Files

| File                        | Purpose                                               |
|-----------------------------|-------------------------------------------------------|
| `train_lora.py`             | LoRA fine-tuning loop on MimicGen clips               |
| `dataset.py`                | full-clip training dataset                            |
| `mimicgen_dataset.py`       | windowed MimicGen frame/prompt dataset                |
| `lora_utils.py`             | inject / save / load LoRA weights                     |
| `inference_lora.py`         | load the i2v model and generate future frames         |
| `run_mimicgen_inference.py` | batch-generate future frames for MimicGen episodes    |
| `run_episode_inference.py`  | generate a full imagined episode rollout              |
| `eval_i2v.py`               | evaluate generation quality                           |
| `config.yaml`               | fine-tuning / inference configuration                 |
| `run_train.sh`              | training entry point                                  |
| `run_inference.sh`          | inference entry point                                 |

## Usage

```bash
bash 1_video_finetuning/run_train.sh                    # fine-tune
python 1_video_finetuning/run_mimicgen_inference.py     # generate future frames
```
