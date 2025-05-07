export HF_HOME="/root/autodl-tmp/huggingface"
export TRANSFORMERS_CACHE="/root/autodl-tmp/huggingface"
# sofa_allen_0507_config_1
accelerate launch flux_train.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --jsonl_for_train="Sofa_test1/metadata_allen.jsonl" \
  --output_dir="sofa_allen_0507_config_1" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --rank=16\
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_prompt_file="validation_prompts_allen.json" \
  --num_validation_images_per_prompt=2 \
  --validation_steps=100 \
  --use_custom_tokens \
  --custom_tokens="<😊1>" \
  --token_init_words="sofa" \
  --save_token_embeddings \
  --seed=0 \