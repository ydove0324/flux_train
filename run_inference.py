#!/usr/bin/env python
# coding=utf-8
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig, set_peft_model_state_dict
import argparse
from PIL import Image
import json
import datetime
from pathlib import Path
import token_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with FLUX model and LoRA weights")
    parser.add_argument(
        "--base_model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to the base model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="sofa_test1/lora_weights.pt",
        help="Path to the LoRA weights file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Beige single sofa with curved design",
        help="Prompt for image generation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.png",
        help="File path to save the generated image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_outputs",
        help="Directory to save validation images",
    )
    parser.add_argument(
        "--validation_prompt_file",
        type=str,
        default=None,
        help="Path to JSON file containing validation prompts",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=50,
        help="Run validation every N steps",
    )
    parser.add_argument(
        "--validation_images_per_prompt",
        type=int,
        default=5,
        help="Number of images to generate for each validation prompt",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for the model",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Resolution of the output image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of steps to validate",
    )
    parser.add_argument(
        "--use_custom_tokens",
        action="store_true",
        help="Whether to use custom tokens",
    )
    parser.add_argument(
        "--custom_tokens",
        type=str,
        nargs="+",
        default=["<😊1>"],
        help="Custom tokens to add to the tokenizer",
    )
    parser.add_argument(
        "--token_init_words",
        type=str,
        nargs="+",
        default=["sofa"],
        help="Words to initialize token embeddings with",
    )
    parser.add_argument(
        "--token_embeddings_file",
        type=str,
        default=None,
        help="Path to saved token embeddings",
    )
    return parser.parse_args()

def setup_pipeline(args):
    """Set up the FLUX pipeline with model and LoRA weights"""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set dtype
    weight_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load the transformer model
    print(f"Loading transformer from {args.base_model}...")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.base_model,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    
    # Configure LoRA
    transformer_lora_config = LoraConfig(
        r=16,  # rank from the training script
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_bias=False,  # Default from training script unless --use_lora_bias was used
    )
    
    # Add LoRA adapter
    transformer.add_adapter(transformer_lora_config)
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}...")
    lora_state_dict = torch.load(args.lora_path, map_location="cpu")
    incompatible_keys = set_peft_model_state_dict(
        transformer, lora_state_dict, adapter_name="default"
    )
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        print(f"Warning: incompatible keys: {incompatible_keys}")
    
    # Create the pipeline
    print("Creating pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    
    # Handle custom tokens
    if args.use_custom_tokens:
        if args.token_embeddings_file is not None and os.path.exists(args.token_embeddings_file):
            # Load token embeddings from file
            print(f"Loading token embeddings from {args.token_embeddings_file}")
            pipeline, token_ids = token_utils.load_token_embeddings(pipeline, args.token_embeddings_file)
        else:
            # Set up custom tokens with initialization
            assert len(args.custom_tokens) == len(args.token_init_words), \
                "Number of custom tokens must match number of initialization words"
            print(f"Setting up custom tokens: {args.custom_tokens} initialized with: {args.token_init_words}")
            pipeline, token_ids = token_utils.setup_custom_tokens(
                pipeline,
                args.custom_tokens,
                args.token_init_words,
                make_learnable=False  # We don't need them to be learnable for inference
            )
    
    # Move to device
    pipeline = pipeline.to(device)
    
    return pipeline, weight_dtype

def run_validation(pipeline, args, step=0):
    """Run validation by generating images for all prompts in the validation file"""
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"step_{step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load validation prompts
    with open(args.validation_prompt_file, 'r') as f:
        validation_prompts = json.load(f)
    
    print(f"Running validation at step {step} with {len(validation_prompts)} prompts")
    
    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    
    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(validation_prompts):
        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(exist_ok=True)
        
        # Save prompt to text file
        with open(prompt_dir / "prompt.txt", "w") as f:
            f.write(prompt)
        
        print(f"Generating {args.validation_images_per_prompt} images for prompt: '{prompt}'")
        
        for img_idx in range(args.validation_images_per_prompt):
            # If seed is provided, generate different seeds for each image
            if args.seed is not None:
                img_generator = torch.Generator(device=pipeline.device).manual_seed(args.seed + img_idx)
            else:
                img_generator = None
                
            # Generate image
            with torch.autocast(pipeline.device.type, dtype=pipeline._dtype):
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=img_generator,
                    max_sequence_length=512,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            
            # Save image
            image_path = prompt_dir / f"image_{img_idx:02d}.png"
            image.save(image_path)
            print(f"  Saved image to {image_path}")
    
    print(f"Validation complete for step {step}. Images saved to {output_dir}")
    return output_dir

def main():
    args = parse_args()
    
    # Set up pipeline
    pipeline, weight_dtype = setup_pipeline(args)
    
    # Single image generation mode
    if args.validation_prompt_file is None:
        # Set up generator for reproducibility
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
        
        # Run inference
        print(f"Generating image with prompt: '{args.prompt}'")
        with torch.autocast(pipeline.device.type, dtype=weight_dtype):
            image = pipeline(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=512,
                height=args.resolution,
                width=args.resolution,
            ).images[0]
        
        # Save the image
        image.save(args.output_file)
        print(f"Image saved to {args.output_file}")
    
    # Validation mode - generate images for multiple prompts at regular intervals
    else:
        # Run validation at step 0
        run_validation(pipeline, args, step=0)
        
        # Run validation at regular intervals
        for step in range(args.validation_interval, args.max_steps + 1, args.validation_interval):
            run_validation(pipeline, args, step=step)

if __name__ == "__main__":
    main() 