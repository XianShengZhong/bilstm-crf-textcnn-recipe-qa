from models.bilstm_crf import BILSTM_CRF
from pathlib import Path
import torch


class BilstmCrfPredictor:
    """
    BiLSTM-CRF 模型预测器
    用于加载训练好的模型并进行实体识别预测

    参数:
        model_path: 模型文件路径
        device: 运行设备 ('cpu' 或 'cuda')
    """
    def __init__(self, model_path, device='cpu'):
        # 初始化设备
        self.device = torch.device(device)
        # 定义特殊标记和标签映射
        self.unk_token = '<unk>'
        self.tag_to_id = {'O': 0, 'B-cook': 1, 'I-cook': 2, 'E-cook': 3, 'B-cuisine': 4, 'I-cuisine': 5, 'E-cuisine': 6, 'B-feature': 7, 'I-feature': 8, 'E-feature': 9,
                 'B-ingredient': 10, 'I-ingredient': 11,  'E-ingredient': 12, '<START>': 13, '<STOP>': 14}
        self.id_to_tag = [tag for tag in self.tag_to_id.keys()]
        # 读取原始数据并构建词汇表
        self.sentens = self._read_raw()
        self.word_to_id = self._build_vocab(self.sentens)
        # 加载模型
        self.model = BILSTM_CRF.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _read_raw(self):
        """
        读取原始文本数据
        返回:
            句子列表
        """
        sentences = []
        file_path = Path(__file__).parent.parent / 'data' / 'cookbook_unhandled.txt'
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                elif len(line) != 0:
                    sentences.append(line)
        return sentences

    def _build_vocab(self, sentences):
        """
        构建词汇表
        参数:
            sentences: 句子列表
        返回:
            单词到索引的映射字典
        """
        word_to_ix = {self.unk_token: 0}
        for sentence in sentences:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix

    def prepare_sequence(self, seq):
        """
        将单词序列转换为索引序列
        参数:
            seq: 输入单词序列
        返回:
            对应的索引张量
        """
        idxs = [self.word_to_id.get(w, self.word_to_id[self.unk_token]) for w in seq]
        return torch.tensor(idxs, dtype=torch.long)


if __name__ == '__main__':
    # 使用示例
    predictor = BilstmCrfPredictor('model_save/bilstm_crf_model.pth')
    id_to_tag = predictor.id_to_tag

    # 示例预测
    texts = ["这道湘菜的豆苗炒虾片非常有名，它的特点是烧，主要用到虾仁、胡萝卜", "请问做宫保鸡丁的核心食材是什么？", '我想做一道酸甜美味的菜，请问怎么做', '请问宫保鸡丁需要哪些食材？']
    for text in texts:
        _, result = predictor.model(predictor.prepare_sequence(text))
        print(f"文本: {text}")
        print([id_to_tag[tag] for tag in result])
        # print(f"实体识别结果: {result}")
        # print(type(result))
        print("-" * 40)