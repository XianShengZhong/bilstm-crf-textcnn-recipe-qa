from models.textcnn import TEXTCNN
import torch


class TextCnnPredictor:
    """TextCNN模型预测器，用于文本分类任务"""
    def __init__(self, model_path,  device='cpu'):
        """
        初始化预测器
        参数:
            model_path: 预训练模型路径
            device: 指定运行设备，默认使用CPU
        """
        self.device = torch.device(device)
        self.model, self.word_to_id, self.tag_to_id, self.max_len = TEXTCNN.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()


        # 添加规则过滤
    def _is_valid_input(self, text):
       # 基础规则：至少包含动词+名词
       query_words = ["做", "制作", "烹饪", "烧", "煮", "炒", "怎么", "如何", "请教", "推荐", "有什么", "是什么", "属于什么", "有哪些", "需要哪些", "哪些", "哪种"]
       return  any(q in text for q in query_words)

        # 预测函数
    def predict(self, text):
        """
               执行文本分类预测

               参数:
                   text: 待分类的输入文本

               返回:
                   str: 预测类别标签或"无效输入"提示
        """
        if not self._is_valid_input(text):
            return "无效输入"
        self.model.eval()
        # 预处理输入文本
        seq = [self.word_to_id.get(word, self.word_to_id["<unk>"]) for word in text]
        if len(seq) < self.max_len:
            seq += [self.word_to_id["<pad>"]] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]

        tensor = torch.LongTensor(seq).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(1).item()

        return list(self.tag_to_id.keys())[pred]


if __name__ == '__main__':
    predictor = TextCnnPredictor('model_save/textcnn_model.pth')
    texts = ['有什么湘菜可以推荐吗？', '宫保鸡丁怎么做？', '你还好吗', '请问做宫保鸡丁需要用到哪些原料？']
    for text in texts:
        result = predictor.predict(text)
        print(f'文本:{text}')
        print(f'结果:{result}')

