export HF_HOME="/root/autodl-tmp/huggingface"
export TRANSFORMERS_CACHE="/root/autodl-tmp/huggingface"
accelerate launch flux_train.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --jsonl_for_train="Sofa_test1/metadata.jsonl" \
  --output_dir="sofa_test2" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --rank=16\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt_file="validation_prompts.json" \
  --num_validation_images_per_prompt=2 \
  --validation_steps=50 \
  --use_custom_tokens \
  --custom_tokens="<😊1>" \
  --token_init_words="sofa" \
  --save_token_embeddings \
  --seed=0 \