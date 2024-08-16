import os
import json
import torch
import random
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoProcessor

ratio = 0.9
epochs = 1
batch_size = 6
learning_rate = 1e-5
accumulation_steps = 4
model_name = "microsoft/Florence-2-base-ft"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "florence_captcha_dataset_full.json"
model_save_path = "models"


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


def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs, answers


model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, revision="refs/pr/6"
).to(device)
processor = AutoProcessor.from_pretrained(
    model_name, trust_remote_code=True, revision="refs/pr/6"
)

for param in model.vision_tower.parameters():
    param.is_trainable = False

with open(data_path, "r") as f:
    data = json.load(f)

random.shuffle(data)

train_dataset = QureDataset(data)
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset,
    [
        int(ratio * len(train_dataset)),
        len(train_dataset) - int(ratio * len(train_dataset)),
    ],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
)

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = epochs * len(train_loader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

scaler = GradScaler()

for epoch in range(epochs):
    model.train()
    train_loss = 0
    optimizer.zero_grad()

    for step, (inputs, answers) in enumerate(
        tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
    ):
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        labels = processor.tokenizer(
            text=answers, return_tensors="pt", padding=True, return_token_type_ids=False
        ).input_ids.to(device)

        with autocast():
            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for step, (inputs, answers) in enumerate(
            tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}")
        ):
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_train_loss = train_loss / len(train_loader)
    print(
        f"Epoch: {epoch+1} | Average Training Loss: {avg_train_loss} | Average Validation Loss: {avg_val_loss}"
    )

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    output_dir = f"{model_save_path}/epoch_{epoch+1}_{avg_val_loss}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
