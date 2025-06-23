from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# 读取数据集
def read_data():
    """
    读取训练数据文件
    返回:
        list of tuples: [(label, text), ...]
    """
    data = []
    file_path = Path(__file__).parent.parent / 'data' / 'TEXTCNN_train.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line)  == 0:
                continue
            lable, senten = line.strip().split('    ')
            data.append((lable, senten))
    return data
data  = read_data()
# print(data)

# 构建词汇表
def build_vocab(texts):
    """
    构建词汇表
    参数:
        texts: 所有文本列表
    返回:
        dict: 词汇到索引的映射
    """
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for word in text:
            if word not in vocab.keys():
                vocab[word] = len(vocab)-1
    return vocab


class TextDataset(Dataset):
    """
    文本数据集类
    实现文本数据的加载和预处理
    """
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 文本转为索引序列
        seq = [self.word_to_idx.get(word, self.word_to_idx["<unk>"])
               for word in text]
        # 填充/截断
        if len(seq) < self.max_len:
            seq += [self.word_to_idx["<pad>"]] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]

        return torch.LongTensor(seq), torch.LongTensor([label])
# TextCNN模型实现
class TEXTCNN(nn.Module):
    """
    TextCNN 模型实现
    用于文本分类任务
    """
    def __init__(self, vocab_size, embed_dim, num_classes,
                 filter_sizes=[3, 4, 5], num_filters=100):
        super(TEXTCNN, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 多个并行卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim))
            for fs in filter_sizes
        ])

        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 嵌入层
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1)    # (batch_size, 1, seq_len, embed_dim)

        # 通过每个卷积层并应用ReLU
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, seq_len - fs + 1, 1)
            conv_out = conv_out.squeeze(3)  # (batch_size, num_filters, seq_len - fs + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)

        # 拼接所有卷积层的输出
        x = torch.cat(conv_outputs, 1)  # (batch_size, len(filter_sizes)*num_filters)
        x = self.dropout(x)
        logits = self.fc(x)  # (batch_size, num_classes)

        return logits

    @classmethod
    def load_model(cls, path, device='cpu'):
        """加载保存的模型"""
        checkpoint = torch.load(path, map_location=device)

        # 从checkpoint获取模型参数
        params = checkpoint['model_params']
        model = cls(
            vocab_size=params['vocab_size'],
            embed_dim=params['embed_dim'],
            num_classes=params['num_classes'],
            filter_sizes=params['filter_sizes'],
            num_filters=params['num_filters']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model, checkpoint['word_to_idx'], checkpoint['tag_to_id'], params['max_len']


def train(model, iterator, optimizer, criterion):
   """
   训练函数
   参数:
       model: 模型
       iterator: 数据迭代器
       optimizer: 优化器
       criterion: 损失函数
   返回:
       tuple: (平均损失, 平均准确率)
   """
   model.train()
   epoch_loss = 0
   epoch_acc = 0

   for batch in tqdm(iterator):
       text, labels = batch
       text, labels = text.to(device), labels.to(device).squeeze(1)

       optimizer.zero_grad()
       predictions = model(text)
       loss = criterion(predictions, labels)
       acc = (predictions.argmax(1) == labels).float().mean()

       loss.backward()
       optimizer.step()

       epoch_loss += loss.item()
       epoch_acc += acc.item()

   return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 添加规则过滤
def is_valid_input(text):
    """
    输入有效性检查
    参数:
        text: 输入文本
    返回:
        bool: 是否有效
    """
    # 基础规则：至少包含动词+名词
    query_words = ["做", "制作", "烹饪", "烧", "煮", "炒", "怎么", "如何", "请教", "推荐", "有什么", "是什么", "属于什么", "有哪些", "需要哪些", "哪些", "哪种"]
    return any(q in text for q in query_words)
    # 预测函数


def predict(text, model, word_to_idx, max_len, device):
    """
    预测函数
    参数:
        text: 输入文本
        model: 模型
        word_to_idx: 词汇表
        max_len: 最大长度
        device: 设备
    返回:
        str: 预测结果
    """
    if not is_valid_input(text):
        return "无效输入"
    model.eval()
    # 预处理输入文本
    seq = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
    if len(seq) < max_len:
        seq += [word_to_idx["<pad>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()
    return list(tag_to_id.keys())[pred]

# 准备数据
data = read_data()
tag_to_id = {'to do': 0, 'to recommend': 1, 'to find cuisine': 2, 'to find feature': 3, 'to find ingredient': 4}
texts = [text for _, text in data]
labels = [tag_to_id[label] for label, _ in data]  # 转为id
word_to_idx = build_vocab(texts)
vocab_size = len(word_to_idx)
max_len = max(len(text) for text in texts)  # 或设定固定长度
batch_size = 32
train_dataset = TextDataset(texts, labels, word_to_idx, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
embed_dim = 128

if __name__ == '__main__':
    # 初始化模型
    model = TEXTCNN(vocab_size, embed_dim, num_classes=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

    model_save_path = Path(__file__).parent / 'model_save' / 'textcnn_model.pth'
    model_save_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_idx': word_to_idx,
        'tag_to_id': tag_to_id,
        'model_params': {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'num_classes': 5,
            'filter_sizes': [3, 4, 5],
            'num_filters': 100,
            'max_len': max_len
        }
    }, model_save_path)
    print(f"模型已保存到 {model_save_path}")

    while True:
        test_text = input("请输入: ")
        print(predict(test_text, model, word_to_idx, max_len, device))



