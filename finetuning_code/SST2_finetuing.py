import torch
from transformers import T5Tokenizer, T5ForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

def train_model(model, train_dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_dataloader), desc="Training") as pbar:
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.update(1)

    average_loss = total_loss / len(train_dataloader)
    return average_loss

def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    with tqdm(total=len(val_dataloader), desc="Validation") as pbar:
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            val_correct += torch.sum(preds == labels).item()
            pbar.update(1)

    val_accuracy = val_correct / len(val_dataloader.dataset)
    return val_loss, val_accuracy

def train_and_validate(model_name, train_texts, train_labels, val_texts, val_labels, epochs=3, lr=2e-5, save_path= None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train_model(model, train_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

        val_loss, val_accuracy = validate_model(model, val_dataloader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if save_path is not None:
            model_save_path = os.path.join(save_path, f"t5_model_add_validation_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at: {model_save_path}")


dataset = load_dataset("glue", "sst2")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset["train"]["sentence"], dataset["train"]["label"], test_size=0.2, random_state=42
)

base_path = os.getcwd()
save_path = os.path.join(base_path,'finetuning/SST2/')

train_and_validate("t5-base", train_texts, train_labels, val_texts, val_labels, epochs=10, lr=1e-4, save_path=save_path)
