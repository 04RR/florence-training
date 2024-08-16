import argparse
import os
from functools import partial
import random
import json

import friendlywords as fw
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

import wandb
from torch.cuda.amp import autocast, GradScaler
from peft import LoraConfig, get_peft_model
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class QureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        question = item["prefix"]
        answer = item["suffix"]

        image_path = item["image"]
        image = Image.open(image_path).convert("RGB")

        return question, answer, image


def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True, truncation=True, max_length=800
    ).to(device)
    return inputs, answers


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
):
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size // 2,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=val_sampler,
    )

    return train_loader, val_loader


def evaluate_model(rank, world_size, model, val_loader, device, train_loss, processor, global_step, batch_size, max_val_item_count):
    if rank == 0:
        avg_train_loss = train_loss / (global_step * batch_size * world_size)
        wandb.log({"step": global_step, "train_loss": avg_train_loss})
        print(f"Rank {rank} - Average Training Loss: {avg_train_loss}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_item_count = 0
        for batch in tqdm(val_loader, desc=f"Evaluation at step {global_step}", position=rank):
            val_item_count += len(batch)
            inputs, answers = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

            val_loss += loss.item()
            if val_item_count > max_val_item_count:
                break

    avg_val_loss = val_loss / val_item_count
    print(f"Rank {rank} - Step {global_step} - Average Validation Loss: {avg_val_loss}")

    if rank == 0:
        wandb.log({"val_loss": avg_val_loss, "step": global_step})

    model.train()


def train_model(rank, world_size, data_path, batch_size=6, use_lora=False, epochs=10, lr=1e-6, eval_steps=10, run_name=None, max_val_item_count=1000):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    if rank == 0:
        wandb.init(project="DocVQA-instruct", name=run_name)
        wandb.config.update({
            "batch_size": batch_size,
            "use_lora": use_lora,
            "epochs": epochs,
            "learning_rate": lr,
            "eval_steps": eval_steps,
            "world_size": world_size,
        })

    with open(data_path, "r") as f:
        data = json.load(f)

    random.shuffle(data)

    ratio = 0.9
    train_dataset = QureDataset(data)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [
            int(ratio * len(train_dataset)),
            len(train_dataset) - int(ratio * len(train_dataset)),
        ],
    )

    model = AutoModelForCausalLM.from_pretrained(
        "andito/Florence-2-large-ft", trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "andito/Florence-2-large-ft", trust_remote_code=True
    )

    if use_lora:
        TARGET_MODULES = [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "linear", "Conv2d", "lm_head", "fc2"
        ]

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)

    model = DDP(model, device_ids=[rank])

    num_workers = 0
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size,
        num_workers,
        rank,
        world_size,
        processor,
        device,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank
        ):
            inputs, answers = batch

            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            global_step += 1

            if global_step % eval_steps == 0:
                evaluate_model(rank, world_size, model, val_loader, device, train_loss, processor, global_step, batch_size, max_val_item_count)

        evaluate_model(rank, world_size, model, val_loader, device, train_loss, processor, global_step, batch_size, max_val_item_count)

        avg_train_loss = train_loss / len(train_loader)
        if rank == 0:
            wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss})

        if rank == 0:
            output_dir = f"./model_checkpoints/{run_name}/epoch_{epoch + 1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    if rank == 0:
        wandb.finish()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train Florence-2 model on specified dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the JSON dataset")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--use-lora", action='store_true', help="Use LoRA if this flag is passed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--max-val-item-count", type=int, default=1000, help="Maximum number of items to evaluate on during validation")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    for rank in range(world_size):
        train_model(rank, world_size, args.data_path, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, args.run_name, args.max_val_item_count)


if __name__ == "__main__":
    main()
