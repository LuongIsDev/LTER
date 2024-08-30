import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("phobert_lter_model").to(device)


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

class LegalTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        statement = item['statement']
        label = int(item['original_label'] == 'Yes')  # Chuyển đổi nhãn thành số (0 hoặc 1)

        encoding = self.tokenizer(statement, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Hàm đánh giá mô hình
def evaluate_model(test_loader, model):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Tính toán báo cáo phân loại
    report = classification_report(all_labels, all_preds, target_names=['No', 'Yes'])
    print("Classification Report:\n", report)
    
    # Tính toán độ chính xác tổng thể
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Tính toán ma trận nhầm lẫn
    cm = confusion_matrix(all_labels, all_preds)
    return cm

def plot_train_loss():
    with open('train_loss.json', 'r') as f:
        data = json.load(f)
        train_losses = data['train_losses']
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def main():
    test_data = load_data('processed_test.json')
    
    test_dataset = LegalTextDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Đánh giá mô hình
    cm = evaluate_model(test_loader, model)

    # Vẽ biểu đồ loss
    plot_train_loss()

    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()
