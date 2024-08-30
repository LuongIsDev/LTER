import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import numpy as np

# Khởi tạo thiết bị (GPU hoặc CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải PhoBERT tokenizer và mô hình cho phân loại
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=2).to(device)

# Hàm tải dữ liệu từ file JSON
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# Tạo dataset cho PyTorch
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

# Hàm huấn luyện mô hình
def train_model(train_loader, model, optimizer, num_epochs=10):
    model.train()
    train_losses = []  # Danh sách lưu loss trong mỗi epoch
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)  # Lưu giá trị loss
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

    # Lưu loss vào file JSON
    with open('train_loss.json', 'w') as f:
        json.dump({'train_losses': train_losses}, f)

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
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy * 100:.2f}%')

def main():
    # Tải dữ liệu
    train_data = load_data('processed_train.json')
    test_data = load_data('processed_test.json')
    
    # Tạo Dataset và DataLoader
    train_dataset = LegalTextDataset(train_data, tokenizer)
    test_dataset = LegalTextDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Khởi tạo optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Huấn luyện mô hình
    train_model(train_loader, model, optimizer, num_epochs=25)
    
    # Đánh giá mô hình
    evaluate_model(test_loader, model)

    # Lưu mô hình
    model.save_pretrained('phobert_lter_model')
    tokenizer.save_pretrained('phobert_lter_model')

if __name__ == "__main__":
    main()
