import torch
import optuna
from pathlib import Path
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def read_data():
    """
    读取训练数据并解析为句子和标签列表
    返回:
        sentences: 二维列表，每个子列表是一个句子的单词序列
        tags: 二维列表，每个子列表是对应的标签序列
    """
    sentences = []
    tags = []
    file_path = Path(__file__).parent.parent / 'data' / 'BILSTM_CRF_train.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        tmp_sentence = []
        tmp_tags = []
        for line in f:
            if line == '  O\n' and len(tmp_sentence) != 0:
                assert len(tmp_sentence) == len(tmp_tags)
                sentences.append(tmp_sentence)
                tags.append(tmp_tags)
                tmp_sentence = []
                tmp_tags = []
            else:
                line = line.strip().split(' ')
                if len(line) == 2:
                    tmp_sentence.append(line[0])
                    tmp_tags.append(line[1])
        if len(tmp_sentence) != 0:
            assert len(tmp_sentence) == len(tmp_tags)
            sentences.append(tmp_sentence)
            tags.append(tmp_tags)
    return sentences, tags


def read_raw():
    """读取原始文本数据"""
    sentences = []
    file_path = Path(__file__).parent.parent / 'data' / 'cookbook_unhandled.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                continue
            elif len(line) != 0:
                sentences.append(line)
    return sentences


def build_vocab(sentences):
    """构建词汇表"""
    global word_to_ix
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def prepare_sequence(seq, to_ix):
    """将单词序列转换为索引序列"""
    idxs = [to_ix.get(w, to_ix[unk_token]) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def argmax(vec):
    # 得到最大的值的索引
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    """计算log-sum-exp防止数值溢出"""
    max_score = vec[0, argmax(vec)]  # max_score的维度为1
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # 维度为1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BILSTM_CRF(nn.Module):
    """BiLSTM-CRF模型实现"""

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout=0.5):
        super(BILSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.dropout = nn.Dropout(dropout)  # 添加dropout层
        # 转移矩阵，transitions[i][j]表示从label_j转移到label_i的概率,虽然是随机生成的但是后面会迭代更新
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 从任何标签转移到START_TAG不可能
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # 从STOP_TAG转移到任何标签不可能

        # 新增约束：O标签不能转移到B/I/E标签
        for tag in ['I-cook', 'E-cook', 'I-cuisine', 'E-cuisine', 'I-feature', 'E-feature', 'I-ingredient', 'E-ingredient']:
            self.transitions.data[tag_to_ix[tag], tag_to_ix['O']] = -10000
            self.transitions.data[tag_to_ix[tag], tag_to_ix[START_TAG]] = -10000

            # 强制B-X后必须接I-X或E-X
        for tag_type in ['cook', 'cuisine', 'feature', 'ingredient']:
                b_tag = f'B-{tag_type}'
                i_tag = f'I-{tag_type}'
                e_tag = f'E-{tag_type}'
                tags = [b_tag,i_tag,e_tag]
                # 禁止B-X后接其他类型的I/E
                for other_type in ['cook', 'cuisine', 'feature', 'ingredient']:
                    if other_type != tag_type:
                        for tag in tags:
                            self.transitions.data[tag_to_ix[f'B-{other_type}'], tag_to_ix[tag]] = -10000
                            self.transitions.data[tag_to_ix[f'I-{other_type}'], tag_to_ix[tag]] = -10000
                            self.transitions.data[tag_to_ix[f'E-{other_type}'], tag_to_ix[tag]] = -10000

        self.hidden = self.init_hidden()  # 随机初始化LSTM的输入(h_0, c_0)

    def init_hidden(self):
        """初始化LSTM隐藏状态"""
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        '''
        输入：发射矩阵(emission score)，实际上就是LSTM的输出——sentence的每个word经BiLSTM后，对应于每个label的得分
        输出：所有可能路径得分之和/归一化因子/配分函数/Z(x)
        '''
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 包装到一个变量里面以便自动反向传播
        forward_var = init_alphas
        for feat in feats:  # w_i
            alphas_t = []
            for next_tag in range(self.tagset_size):  # tag_j
                # t时刻tag_i emission score（1个）的广播。需要将其与t-1时刻的5个previous_tags转移到该tag_i的transition scors相加
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)  # 1*5
                # t-1时刻的5个previous_tags到该tag_i的transition scors
                trans_score = self.transitions[next_tag].view(1, -1)  # 维度是1*5

                next_tag_var = forward_var + trans_score + emit_score
                # 求和，实现w_(t-1)到w_t的推进
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)  # 1*5

        # 最后将最后一个单词的forward var与转移 stop tag的概率相加
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        '''
        输入：id化的自然语言序列
        输出：序列中每个字符的Emission Score
        '''
        self.hidden = self.init_hidden()  # (h_0, c_0)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_out = self.dropout(lstm_out)  # 添加dropout
        lstm_feats = self.hidden2tag(lstm_out)  # len(s)*5
        return lstm_feats

    def _score_sentence(self, feats, tags):
        '''
        输入：feats——emission scores；tags——真实序列标注，以此确定转移矩阵中选择哪条路径
        输出：真实路径得分
        '''
        score = torch.zeros(1)
        # 将START_TAG的标签３拼接到tag序列最前面
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 预测序列的得分，维特比解码，输出得分与路径值
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]  # forward_var保存的是之前的最优路径的值
                best_tag_id = argmax(next_tag_var)  # 返回最大值对应的那个tag
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # bptrs_t有５个元素

        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 无需返回最开始的START位
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):  # 损失函数
        feats = self._get_lstm_features(sentence)  # len(s)*5
        forward_score = self._forward_alg(feats)  # 规范化因子/配分函数
        gold_score = self._score_sentence(feats, tags)  # 正确路径得分
        return forward_score - gold_score  # Loss（已取反）


    def forward(self, sentence):
        '''
        解码过程，维特比解码选择最大概率的标注路径
        '''
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def save_model(self, path):
        """保存模型函数"""
        torch.save({
            'vocab_size': self.vocab_size,
            'tag_to_ix': self.tag_to_ix,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout.p,  # 保存dropout率
            'transitions': self.transitions.data,  # 显式保存转移矩阵
            'model_state_dict': self.state_dict()
        }, path)

    @classmethod
    def load_model(cls, path):
        """加载模型函数"""
        checkpoint = torch.load(path)
        model = cls(
            checkpoint['vocab_size'],
            checkpoint['tag_to_ix'],
            checkpoint['embedding_dim'],
            checkpoint['hidden_dim'],
            checkpoint['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.transitions.data = checkpoint['transitions']
        return model

def train_with_optuna():
    # 定义Optuna目标函数
    def objective(trial):
        # 设置可调参数
        embedding_dim = trial.suggest_categorical('embedding_dim', [50, 100, 150])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 0.001, 0.1, log=True)

        # 创建模型
        model = BILSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim, dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        best_model = None  # 用于保存当前trial的最佳模型
        best_loss = float('inf')

        # 训练循环
        num = 0
        for epoch in range(3):  # 保持原有epoch数
            num += 1
            print(f'第{num}次训练')
            total_loss = 0
            for sentence, tags in zip(train_sentens, train_tags):
                model.zero_grad()
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

                loss = model.neg_log_likelihood(sentence_in, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 输出每轮平均损失
            avg_loss = total_loss / len(train_sentens)
            print(f'平均损失{avg_loss}')
            trial.report(avg_loss, epoch)

            # 提前停止机制
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

                # 保存当前trial中表现最好的模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = BILSTM_CRF(
                    len(word_to_ix),
                    tag_to_ix,
                    embedding_dim,
                    hidden_dim,
                    dropout
                )
                best_model.load_state_dict(model.state_dict())
                best_model.transitions.data = model.transitions.data.clone()  # 保存转移矩阵
        # 保存最佳模型到trial
        trial.set_user_attr("best_model_state", best_model.state_dict())
        trial.set_user_attr("best_transitions", best_model.transitions.data.clone())
        return best_loss

    # 创建并运行Optuna研究
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3)  # 运行3次试验

    # 获取最佳参数重建模型
    best_trial = study.best_trial
    final_model = BILSTM_CRF(
        len(word_to_ix),
        tag_to_ix,
        best_trial.params['embedding_dim'],
        best_trial.params['hidden_dim'],
        best_trial.params['dropout']
    )

    # 加载最佳状态
    final_model.load_state_dict(best_trial.user_attrs['best_model_state'])
    final_model.transitions.data = best_trial.user_attrs['best_transitions']

    # 保存模型
    model_dir = Path(__file__).parent / "model_save"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "bilstm_crf_model.pth"
    final_model.save_model(model_path)
    print(f"模型已保存到: {model_path}")

    return final_model


unk_token = '<unk>'
START_TAG = "<START>"
STOP_TAG = "<STOP>"
word_to_ix = {unk_token: 0}
sentens = read_raw()
word_to_ix = build_vocab(sentens)
train_sentens, train_tags = read_data()
tag_to_ix = {'O': 0, 'B-cook': 1, 'I-cook': 2, 'E-cook': 3, 'B-cuisine': 4, 'I-cuisine': 5, 'E-cuisine': 6,
             'B-feature': 7, 'I-feature': 8, 'E-feature': 9,
             'B-ingredient': 10, 'I-ingredient': 11, 'E-ingredient': 12, START_TAG: 13, STOP_TAG: 14}
ix_to_tag = [tag for tag in tag_to_ix.keys()]

if __name__ == "__main__":

    # 训练模型
    model = train_with_optuna()

    # 交互式预测
    with torch.no_grad():
        while True:
            a = input("请输入:")
            precheck_sent = prepare_sequence(a, word_to_ix)
            _, tags = model(precheck_sent)
            print([ix_to_tag[tag] for tag in tags])
