export DEEPSPEED_ACTIVATION_CHECKPOINTING=0
export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTENTION=true
export CUTLASS_PATH=/home/psi-cmd/projects/Protenix/cutlass_v3.3.0
export CUDA_LAUNCH_BLOCKING=1

checkpoint_path="./release_data/checkpoint/model_v0.5.0.pt"

python3 ./runner/train.py \
--run_name molecular_glue_finetune \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb false \
--diffusion_batch_size 48 \
--diffusion_chunk_size 12 \
--eval_interval 200 \
--log_interval 20 \
--checkpoint_interval 1000 \
--ema_decay 0.999 \
--train_crop_size -1 \
--max_steps 5000 \
--warmup_steps 200 \
--lr 3e-4 \
--sample_diffusion.N_step 10 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets molecular_glue_finetune
