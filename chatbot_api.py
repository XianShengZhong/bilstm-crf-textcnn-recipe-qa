from flask import Flask, request
import json
from question_classifier import *
from question_parser import *
from answer_search import *
from flask_cors import CORS

server = Flask(__name__)
CORS(server, resources=r'/*')
classifier = QuestionClassifier()
parser = QuestionPaser()
searcher = AnswerSearcher()
@server.route('/index',methods=['get'])
def index():
    """
       问答系统主接口
       处理流程:
           1. 接收用户问题
           2. 问题分类
           3. 解析生成查询语句
           4. 执行查询获取答案
       """
    res = {}
    # 默认回答
    answer = '对不起呢(ಥ﹏ಥ)，布布还没弄明白，请再问一遍，或者布布真的得去请教一二宝了╥﹏╥'

    # 检查请求参数
    if request.args is None:
        res['code'] = '5004'
        res['info'] = '请求参数为空'
        return json.dumps(res, ensure_ascii=False)

    # 获取请求参数
    param = request.args.to_dict()
    sent = param.get('sent')

    # 问题分类处理
    res_classify = classifier.classify(sent)
    if res_classify == None:
        res['answer'] = answer
        return json.dumps(res, ensure_ascii = False)

    # 解析问题生成查询语句
    res_parser_sql = parser.parser_main(res_classify)

    # 执行查询获取最终答案
    final_answers = searcher.search_main(res_parser_sql)
    res['answer'] = final_answers

    # 返回JSON格式响应
    return json.dumps(res, ensure_ascii = False)


if __name__ == '__main__':
    # 配置JSON响应不转义中文字符
    server.config['JSON_AS_ASCII'] = False
    # 启动服务，监听5001端口
    server.run(port=5001,debug=False)