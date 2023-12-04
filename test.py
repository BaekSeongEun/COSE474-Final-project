import torch
from transformers import T5Tokenizer, T5ForSequenceClassification
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        encoding = self.tokenizer(self.texts.iloc[idx], return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(self.labels.iloc[idx])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

def test_model(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    with tqdm(total=len(test_dataloader), desc="Test") as pbar:
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            test_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            test_correct += torch.sum(preds == labels).item()
            pbar.update(1)

    test_accuracy = test_correct / len(test_dataloader.dataset)
    return test_loss, test_accuracy

def plot_results(loss_values, accuracy_values,save_path):
    plt.figure(figsize=(12, 6))

    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
    plt.title('Test Loss Over 10 Models')
    plt.xlabel('Model Index')
    plt.ylabel('Loss')

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracy_values) + 1), accuracy_values, marker='o', color='orange')
    plt.title('Test Accuracy Over 10 Models')
    plt.xlabel('Model Index')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv('/home/tjddms9376/DL/test.csv')

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForSequenceClassification.from_pretrained('t5-base', num_labels=2).to(device)
criterion = torch.nn.CrossEntropyLoss()

test_dataset = CustomDataset(dataset['text'], dataset['label'], tokenizer)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


fine_tuning = ['SST2','IMDB', 'IMDB_SST2', 'SST2_IMDB']
base_path = os.getcwd()

for fine in fine_tuning:
    model_path = os.path.join(base_path,f'finetuning/{fine}')
    all_loss_values = []
    all_accuracy_values = []
    for i in range(10):
        model.load_state_dict(torch.load(os.path.join(model_path,f't5_model_add_validation_epoch_{i+1}.pt')))
        loss, accuracy = test_model(model, test_dataloader, criterion, device)
        print(f"loss: {loss}, accuracy: {accuracy}, number : {i+1}")

        all_loss_values.append(loss)
        all_accuracy_values.append(accuracy)
    figure_path = os.path.join(base_path,f'result_figure/{fine}.png')
    plot_results(all_loss_values, all_accuracy_values,figure_path)
