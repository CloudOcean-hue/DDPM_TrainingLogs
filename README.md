# 🧨 DDPM Training from Scratch with Hugging Face Diffusers

This repo documents a **complete, from-scratch DDPM training workflow** built with 🤗 Diffusers.

When I was learning, most libraries and examples stopped at theory or inference and didn’t show the full training loop. **DDPMs train slowly**, so it’s easy to doubt whether things are working. To make the process transparent, I’m publishing the **entire run**—loss curves, intermediate samples, and checkpoints—so others can follow and verify their own training.

---

## 📘 Overview

This project trains a diffusion model (`UNet2DModel`) using the DDPM/DDIM process implemented in `diffusers`.  
It includes everything needed for real training runs:

- Data loading and augmentation (`datasets`, `torchvision`)
- Model setup (`UNet2DModel`)
- Noise scheduler (`DDIMScheduler`)
- Optimizer and learning rate scheduler
- EMA (Exponential Moving Average) for stable updates
- Sampling pipeline (`DDIMPipeline`)
- Logging with TensorBoard
- Distributed and mixed-precision training via `Accelerate`

You can use any image dataset on Hugging Face Datasets or your own local dataset.

---

## ⚙️ Key Components

### UNet2DModel

The core neural network that predicts the **noise residual** from a noisy image and timestep.  
It’s the main backbone in diffusion models like Stable Diffusion.

```python
model = UNet2DModel(
    sample_size=args.resolution,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 1024),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
)
```

---

### DDIMScheduler

Defines how noise is added (forward process) and removed (reverse process).  
Here, we use **DDIM (Deterministic Diffusion Implicit Models)** — a faster and deterministic variant of DDPM.

```python
noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
```

---

### EMA (Exponential Moving Average)

EMA keeps a smoothed version of model weights for more stable training and better image generation quality.

```python
ema_model = EMAModel(model.parameters(), decay=args.ema_max_decay, power=args.ema_power)
```

---

### Accelerate Integration

The `Accelerator` from Hugging Face simplifies distributed training and mixed precision — no need to write your own DDP boilerplate.

```python
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    log_with="tensorboard",
    project_dir=logging_dir,
)
```

---

### Diffusion Process

Noise is added to the clean image using the scheduler, and the model learns to predict that noise back — the key objective of DDPM training.

```python
noisy_images = noise_scheduler.add_noise(clean_images, noise_samples, timesteps)
output = model(noisy_images, timesteps).sample
loss = F.mse_loss(output, noise_samples)
```

---

## 🖼️ Sampling During Training

At each epoch, the EMA-weighted model is used to generate sample images for visual inspection.

```python
pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
images = pipeline(generator=torch.manual_seed(0), batch_size=args.eval_batch_size, num_inference_steps=50).images
```

The results are logged to TensorBoard, and optionally pushed to the Hugging Face Hub.

---

## 🧑‍💻 How to Run

### (Optional) Use a Hugging Face Mirror (for China)

```bash
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
source ~/.bashrc
```

### Run the Training

```bash
python main.py     --dataset huggan/pokemon     --output_dir pokemon-ddpm-fixed     --resolution 64     --train_batch_size 16     --num_epochs 400     --learning_rate 1e-4     --gradient_accumulation_steps 1     --lr_warmup_steps 500     --mixed_precision no
```

### Hardware Used

Training was performed on **a single NVIDIA V100 GPU**.

---

## 📊 Training Records

Training logs are automatically saved with TensorBoard in `--logging_dir logs`.  
You can visualize loss curves and sample images during training:

```bash
tensorboard --logdir logs
```

### Example Observations

- Loss (`MSE`) gradually decreases as the model learns.
- Sample images improve in fidelity after ~100 epochs.
- EMA weights ensure smooth convergence and consistent quality.

If you save your training history, you can include it here:

```
📉 Example loss curve (replace with your actual plot)
🖼️ Example generated samples (epoch snapshots)
```

---

## 💾 Outputs

- Trained model checkpoints in `--output_dir`
- EMA weights stored in the same directory
- Sample images and logs under TensorBoard `logs/`
- Optional model push to the Hugging Face Hub

---

## 🧾 Full Training Script

You can paste or link to your full Python script here (e.g., `example_fixed.py`) for reproducibility.

---

## 💡 Notes

- If dataset loading fails, use another public dataset or a local path.
- For higher resolutions, ensure enough GPU memory or enable gradient checkpointing.
- EMA is strongly recommended — disabling it often harms generation quality.

---

## 📚 References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [DDPM Paper (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [DDIM Paper (Song et al., 2020)](https://arxiv.org/abs/2010.02502)
