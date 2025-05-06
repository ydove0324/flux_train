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
from pathlib import Path
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run validation with FLUX model and LoRA weights")
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
        "--validation_prompt_file",
        type=str,
        required=True,
        help="Path to JSON file containing validation prompts",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_outputs",
        help="Directory to save validation images",
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
        "--images_per_prompt",
        type=int,
        default=5,
        help="Number of images to generate for each prompt",
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
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank parameter (r)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
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
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_bias=False,
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
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Load validation prompts
    with open(args.validation_prompt_file, 'r') as f:
        validation_prompts = json.load(f)
    
    print(f"Loaded {len(validation_prompts)} prompts from {args.validation_prompt_file}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(validation_prompts):
        prompt_dir = output_dir / f"prompt_{prompt_idx:03d}"
        prompt_dir.mkdir(exist_ok=True)
        
        # Save prompt to text file
        with open(prompt_dir / "prompt.txt", "w") as f:
            f.write(prompt)
        
        print(f"Generating {args.images_per_prompt} images for prompt: '{prompt}'")
        
        for img_idx in range(args.images_per_prompt):
            # Generate different seed for each image if seed is provided
            if args.seed is not None:
                generator = torch.Generator(device=device).manual_seed(args.seed + img_idx)
            else:
                generator = None
                
            # Generate image
            with torch.autocast(device, dtype=weight_dtype):
                image = pipeline(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    max_sequence_length=512,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            
            # Save image
            image_path = prompt_dir / f"image_{img_idx:02d}.png"
            image.save(image_path)
            print(f"  Saved image to {image_path}")
    
    print(f"Validation complete. Images saved to {output_dir}")

if __name__ == "__main__":
    main() 