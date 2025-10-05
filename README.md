
🧨 DDPM Training from Scratch with Hugging Face Diffusers

This repository provides a complete, end-to-end training pipeline for a Denoising Diffusion Probabilistic Model (DDPM) using the 🤗 Hugging Face diffusers library.
Unlike most online examples that only cover the theory or inference, this script demonstrates how to actually train a DDPM from scratch, log results, and generate samples at each epoch.

⸻

📘 Overview

This project trains a diffusion model (UNet2D) using the DDPM / DDIM process implemented in diffusers.
It covers everything needed for an actual training run:
	•	Data loading and image augmentation (datasets, torchvision)
	•	Model setup (UNet2DModel)
	•	Noise scheduler (DDIMScheduler)
	•	Optimizer and learning rate scheduler
	•	Exponential Moving Average (EMA) model for stable training
	•	Sampling pipeline (DDIMPipeline)
	•	Logging with TensorBoard
	•	Distributed training via Accelerate

You can use any image dataset available on Hugging Face Datasets, or your own custom dataset.

⸻

⚙️ Key Components Explained

🧩 1. UNet2DModel

The core neural network that predicts the noise residual given a noisy image and timestep.
It’s the main architecture for most diffusion models, consisting of downsampling, attention, and upsampling blocks.

model = UNet2DModel(
    sample_size=args.resolution,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 1024),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
)


⸻

🔁 2. DDIMScheduler

The noise scheduler defines how noise is added (forward process) and removed (reverse process).
Here, we use DDIM (Deterministic Diffusion Implicit Models), a variant of DDPM that allows faster sampling.

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)


⸻

⚖️ 3. EMA (Exponential Moving Average)

EMA keeps a smoothed version of model weights to stabilize training and produce higher-quality samples.

ema_model = EMAModel(model.parameters(), decay=args.ema_max_decay, power=args.ema_power)


⸻

🚀 4. Accelerate Integration

Accelerator from Hugging Face simplifies mixed-precision and distributed training — no manual DDP setup needed.

accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    log_with="tensorboard",
    project_dir=logging_dir,
)


⸻

🧠 5. Forward Diffusion Process

Noise is added to the clean image according to the scheduler, and the model learns to predict that noise back.

noisy_images = noise_scheduler.add_noise(clean_images, noise_samples, timesteps)
output = model(noisy_images, timesteps).sample
loss = F.mse_loss(output, noise_samples)


⸻

🖼️ Sampling During Training

At the end of each epoch, the EMA model generates images using DDIMPipeline for visual inspection.
Images are logged to TensorBoard and optionally pushed to the Hugging Face Hub.

pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
images = pipeline(generator=torch.manual_seed(0), batch_size=args.eval_batch_size, num_inference_steps=50).images


⸻

🧑‍💻 How to Run

1️⃣ (Optional) Use Hugging Face Mirror if you’re in China:

export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
source ~/.bashrc


⸻

2️⃣ Run the Training

python example_fixed.py \
    --dataset huggan/pokemon \
    --output_dir pokemon-ddpm-fixed \
    --resolution 64 \
    --train_batch_size 16 \
    --num_epochs 400 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --lr_warmup_steps 500 \
    --mixed_precision no


⸻

3️⃣ Hardware Used

Training was done on a single NVIDIA V100 GPU.

⸻

📊 Logging and Outputs
	•	Loss curves and generated samples are logged to TensorBoard (--logging_dir logs).
	•	After each epoch:
	•	EMA weights are copied to the model.
	•	The pipeline generates a batch of images.
	•	The images are uploaded or saved locally.
	•	Final models are saved under --output_dir.

⸻

🧠 Example Results

During training, you should observe:
	•	Gradual reduction in MSE loss
	•	Increasing visual fidelity in sampled images
	•	Smooth convergence with EMA weights

⸻

🧾 Full Training Script

<insert full code here>

(Paste your full training code here so others can directly run or adapt it.)

⸻

💡 Notes
	•	If you encounter dataset loading issues, use --dataset with another public dataset name or a local path.
	•	For larger resolutions, increase GPU memory or use gradient checkpointing.
	•	EMA is crucial — disabling it often degrades generation quality.

⸻

📚 References
	•	Hugging Face Diffusers
	•	DDPM Paper (Ho et al., 2020)
	•	DDIM Paper (Song et al., 2020)

⸻

Would you like me to insert the full training code directly into the README (formatted and syntax-highlighted), or keep it as a collapsible section to keep the README shorter and cleaner?
