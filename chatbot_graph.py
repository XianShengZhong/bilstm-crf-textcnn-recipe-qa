from question_classifier import *
from question_parser import *
from answer_search import *

class ChatBotGraph:
    """
    问答系统主类
    功能：
        1. 初始化各处理模块
        2. 处理用户输入的问题
        3. 返回系统生成的回答
    """

    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        """
        问答系统主处理方法
        参数:
            sent: 用户输入的问题文本
        返回:
            系统生成的回答文本
        """
        # 默认回答
        answer = '对不起，布布还没弄懂，请再问一遍，或者布布真的还得去增加阅历(ಥ﹏ಥ)'

        # 第一步：问题分类
        res_classify = self.classifier.classify(sent)
        if res_classify == None:
             return answer

        # 第二步：解析问题生成查询语句
        res_parser_sql = self.parser.parser_main(res_classify)

        # 第三步：执行查询获取最终答案
        final_answers = self.searcher.search_main(res_parser_sql)
        return final_answers

if __name__ == '__main__':
    handler = ChatBotGraph()
    while 1:
        question = input("咨询:")
        answer = handler.chat_main(question)
        print('布布:', answer)