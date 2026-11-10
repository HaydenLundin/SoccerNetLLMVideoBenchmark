"""
Step 2: 8 GB VRAM QLoRA Fine-Tuning Harness for Qwen3.5-4B Unified Specialist.
Absorbing LFM2.5-VL-3B and MiniCPM-V 4.6 domain capabilities into a single unified model.

Hardware & Memory Guardrails:
- Base Model: Qwen/Qwen3.5-4B in 4-bit NF4 (BitsAndBytes)
- Adapter: PEFT QLoRA (r=16, lora_alpha=32, lora_dropout=0.05)
  Target modules: q_proj, k_proj, v_proj, o_proj (SDPA attention projection layers)
- Gradient Checkpointing: Enabled with input_require_grads
- Vision Tower: Parameters frozen (0 activation backward memory)
- Optimizer: bitsandbytes.optim.PagedAdamW8bit (lr=2e-4, weight_decay=0.01)
- VRAM ceiling: Strictly < 6.8 GiB on 8 GB RTX 3000 Ada under Windows WDDM
- Slices: 6 uniformly sampled keyframes with burned-in wristwatch [00:SS]
- Output: Concise-CoT + timestamp + label + confidence + visibility
"""

import os
import gc
import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import bitsandbytes as bnb

CONCISE_DISTILLATION_PROMPT = (
    "Spot soccer events: output JSON with commentary, timestamp, label, confidence, visibility."
)

class SoccerNetDistillationDataset(Dataset):
    """
    Multimodal dataset yielding 6-keyframe sequences paired with
    Concise-CoT reasoning and zero-lag timestamp targets.
    """
    def __init__(self, manifest_path: Path, processor: Any, workspace_root: Path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data.get("samples", [])
        self.processor = processor
        self.workspace_root = workspace_root
        # Uniformly sample 6 frames from the 16 extracted frames
        self.frame_indices = [0, 3, 6, 9, 12, 15]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        frame_paths = item["frames"]
        frames = []
        for i, p in enumerate(frame_paths):
            if i in self.frame_indices:
                full_p = self.workspace_root / p
                img = Image.open(full_p).convert("RGB")
                frames.append(img)

        # Build concise target JSON
        clean_events = []
        for ev in item.get("events", []):
            clean_events.append({
                "commentary": ev["commentary"],
                "timestamp": ev["timestamp"],
                "label": ev["label"],
                "confidence": ev.get("confidence", 0.95),
                "visibility": ev.get("visibility", "visible")
            })
        target_json = json.dumps(clean_events)

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in frames] + [{"type": "text", "text": CONCISE_DISTILLATION_PROMPT}]
            },
            {
                "role": "assistant",
                "content": target_json
            }
        ]

        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )

        inputs = self.processor(
            text=[full_text],
            images=[frames],
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Identify prompt vs assistant boundary to mask prompt tokens with -100
        assistant_marker = "</think>\n\n"
        full_decoded = self.processor.decode(input_ids)

        if assistant_marker in full_decoded:
            marker_tokens = self.processor.tokenizer.encode(assistant_marker, add_special_tokens=False)
            marker_len = len(marker_tokens)
            split_idx = 0
            for i in range(len(input_ids) - marker_len):
                if input_ids[i:i+marker_len].tolist() == marker_tokens:
                    split_idx = i + marker_len
                    break
            labels[:split_idx] = -100
        else:
            fallback_marker = "<|im_start|>assistant\n"
            marker_tokens = self.processor.tokenizer.encode(fallback_marker, add_special_tokens=False)
            marker_len = len(marker_tokens)
            split_idx = 0
            for i in range(len(input_ids) - marker_len):
                if input_ids[i:i+marker_len].tolist() == marker_tokens:
                    split_idx = i + marker_len
                    break
            labels[:split_idx] = -100

        # Mask padding or vision placeholder tokens
        vision_pad_id = getattr(self.processor.tokenizer, "image_pad_token_id", None)
        if vision_pad_id is not None:
            labels[labels == vision_pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "mm_token_type_ids": inputs["mm_token_type_ids"][0],
            "labels": labels,
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"]
        }


def collate_fn_single(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Single sample collation for batch_size=1 training preserving all multimodal keys."""
    item = batch[0]
    return {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "mm_token_type_ids": item["mm_token_type_ids"].unsqueeze(0),
        "labels": item["labels"].unsqueeze(0),
        "pixel_values": item["pixel_values"],
        "image_grid_thw": item["image_grid_thw"]
    }


def train_qlora_specialist(
    dataset_manifest: str = "data/qlora_distillation/dataset_manifest.json",
    base_model_id: str = "Qwen/Qwen3.5-4B",
    output_dir: str = "models/qwen3.5-4b-soccernet-qlora",
    epochs: int = 3,
    lr: float = 2e-4,
    grad_accum_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest_path = WORKSPACE_ROOT / dataset_manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    out_model_dir = WORKSPACE_ROOT / output_dir
    out_model_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("  Qwen3.5-4B Unified Multi-Specialist QLoRA Training Harness")
    print(f"  Base Model:      {base_model_id} (4-bit NF4)")
    print(f"  LoRA Config:     r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  Optimizer:       PagedAdamW8bit (lr={lr}, grad_accum={grad_accum_steps})")
    print(f"  VRAM Ceiling:    Strictly < 6.8 GiB (Dedicated RTX 3000 Ada)")
    print(f"  Output Dir:      {out_model_dir}")
    print("==========================================================================")

    # 1. Load Processor
    print(f"[*] Loading Processor from {base_model_id}...")
    processor = AutoProcessor.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        local_files_only=True
    )
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 1764
        processor.image_processor.min_pixels = 784

    # 2. Configure 4-bit NF4 Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print(f"[*] Loading Base Model in 4-bit NF4 (VRAM ceiling < 6.8 GiB)...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True
    )

    # 3. Prepare for k-bit Training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Freeze vision encoder
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    frozen_count = sum(p.numel() for n, p in model.named_parameters() if "visual" in n)
    print(f"[+] Frozen Vision Parameters: {frozen_count:,} (0 activation backward memory)")

    # 4. Configure LoRA Adapter
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    vram_init_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"[+] Initial Model VRAM Allocated: {vram_init_gib:.2f} GiB")

    # 5. Build Dataset and DataLoader
    print(f"[*] Loading Distillation Dataset from {manifest_path}...")
    dataset = SoccerNetDistillationDataset(manifest_path, processor, WORKSPACE_ROOT)
    print(f"[+] Total Dataset Samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn_single
    )

    # 6. Configure 8-bit Optimizer
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )

    total_steps = (len(dataset) // grad_accum_steps) * epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # 7. Training Loop
    print("\n[*] Starting QLoRA Fine-Tuning Execution...")
    model.train()
    global_step = 0
    t_start = time.perf_counter()
    overall_peak_vram = 0.0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        step_loss = 0.0
        optimizer.zero_grad()

        print(f"\n--- Epoch {epoch}/{epochs} ---")
        for i, batch in enumerate(dataloader):
            batch_gpu = {
                k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(**batch_gpu)
            loss = outputs.loss / grad_accum_steps

            loss.backward()
            step_loss += loss.item() * grad_accum_steps
            epoch_loss += loss.item() * grad_accum_steps

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                current_lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                if peak_vram > overall_peak_vram:
                    overall_peak_vram = peak_vram

                print(
                    f"  [Step {global_step:03d}/{total_steps:03d}] "
                    f"Batch {i+1:03d}/{len(dataloader):03d} | "
                    f"Loss: {step_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Peak VRAM: {peak_vram:.2f} GiB"
                )
                step_loss = 0.0

            if (i + 1) % 15 == 0:
                torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"[*] Epoch {epoch} Complete | Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        epoch_save_dir = out_model_dir / f"checkpoint-epoch-{epoch}"
        print(f"[*] Saving checkpoint to {epoch_save_dir}...")
        model.save_pretrained(epoch_save_dir)
        processor.save_pretrained(epoch_save_dir)

    # 8. Save Final Unified Model Adapter
    total_training_time_min = (time.perf_counter() - t_start) / 60.0
    print(f"\n[+] Training Complete in {total_training_time_min:.1f} minutes!")
    print(f"[*] Saving Final Trained Adapter to {out_model_dir}...")
    model.save_pretrained(out_model_dir)
    processor.save_pretrained(out_model_dir)

    # Save training metadata
    train_metadata = {
        "base_model": base_model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "epochs": epochs,
        "learning_rate": lr,
        "dataset_manifest": dataset_manifest,
        "total_samples": len(dataset),
        "peak_vram_gib": round(overall_peak_vram, 2),
        "training_time_min": round(total_training_time_min, 2)
    }
    with open(out_model_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, indent=2)

    print(f"[+] QLoRA Specialist Adapter successfully saved to: {out_model_dir}")
    print(f"[+] Peak VRAM: {overall_peak_vram:.2f} GiB | Total Time: {total_training_time_min:.1f} min")
    print("==========================================================================\n")
    return out_model_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Unified Specialist Fine-Tuning")
    parser.add_argument("--dataset", type=str, default="data/qlora_distillation/dataset_manifest.json")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=str, default="models/qwen3.5-4b-soccernet-qlora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    train_qlora_specialist(
        dataset_manifest=args.dataset,
        base_model_id=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        grad_accum_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
