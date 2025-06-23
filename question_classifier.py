import os
from models.Bilstm_Crf_Predictor import BilstmCrfPredictor
from models.TextCnn_Predictor import TextCnnPredictor
from difflib import SequenceMatcher



"""
    对用户输入的文本进行问题分类
"""
class QuestionClassifier:
    """
        初始化：
            加载特征词的路径并把路径下的所有文件里面的所有特征词加载出来放入各自的列表里面，然后在所有全部放入region_words这个元组里面
    """
    def __init__(self):
        """
        初始化分类器
        加载特征词和模型
        """
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 菜名、菜系、食材特征词路径
        self.dish_name_path = os.path.join(cur_dir, 'data/dish_name.txt')
        self.cuisine_path = os.path.join(cur_dir, 'data/cuisines.txt')
        # self.ingredient_path = os.path.join(cur_dir, 'data/ingredient.txt')

        # 菜名、菜系、食材加载特征词
        self.dish_name_words = [i.strip() for i in open(self.dish_name_path, encoding='utf-8') if i.strip()]
        self.cuisine_words = [i.strip() for i in open(self.cuisine_path, encoding='utf-8') if i.strip()]

        # 加载BilstmCrf模型
        self.BilstmCrfPredictor = BilstmCrfPredictor('./models/model_save/bilstm_crf_model.pth')
        self.BilstmCrf_id_to_tag = self.BilstmCrfPredictor.id_to_tag
        print('BilstmCrf model init finished ......')

        # 不能同时直接加载TextCnn模型,会导致资源占取
        # 延迟加载TextCnn模型
        self._textcnn_loaded = False
        self._TextCnnPredictor = None

    @property
    def TextCnnPredictor(self):
        if not self._textcnn_loaded:
            self._TextCnnPredictor = TextCnnPredictor('./models/model_save/textcnn_model.pth')
            self._textcnn_loaded = True
            print('TextCnn model init finished ......')
        return self._TextCnnPredictor



    def classify(self, question):
        """
        主分类方法
        参数:
            question: 用户输入的问题文本
        返回:
            包含分类结果和实体信息的字典
        """

        # 用bilstm_crf模型对问句进行实体识别以及分类。bc_entities_prediction_result的结构类似于：{'dish_name': ['干烧鱼翅'], 'cuisine': [], 'feature': [], 'ingredient': []}
        bc_entities_prediction_result = self.bc_extract_entities(question)

        # 对bilstm_crf实体（仅对cook）进行调整,列如{'un_dish_name': [], 'best_match_dish_name': []}
        correction_dish_name = self.correction_dish_name(bc_entities_prediction_result)

        # 用textcnn模型进行文本分类得到是'to do', 'to recommend','to find cuisine', 'to find feature', 'to find ingredient', '无效输入'其中的一类
        tc_textclassify_result = self.TextCnnPredictor.predict(question)
        result_dict = {}
        question_type = ''

        # 根据不同分类结果进行处理
        # 处理"怎么做"类问题
        if tc_textclassify_result == 'to do':
            question_type = "according to dish_name how to do it"
            result_dict['dish_name'] = []
            result_dict['question_type'] = question_type
            # 提取不存在或者写错的最相似的菜名
            un_index, un_dish_name, best_index, best_match_dish_name = [],[],[],[]
            if correction_dish_name["un_dish_name"]:
                for temple in correction_dish_name['un_dish_name']:
                    un_index.append(temple[0])
                    un_dish_name.append(temple[1])
            if correction_dish_name['best_match_dish_name']:
                for temple in correction_dish_name['best_match_dish_name']:
                    best_index.append(temple[0])
                    best_match_dish_name.append(temple[1])
            # 对bilstm_crf提取的菜名进行分类
            for dish_name in bc_entities_prediction_result['dish_name']:
                index = bc_entities_prediction_result['dish_name'].index(dish_name)
                if (index not in un_index) and (index not in best_index):
                    result_dict['dish_name'].append(bc_entities_prediction_result['dish_name'][index])
            result_dict['un_dish_name'] = un_dish_name
            result_dict['best_match_dish_name'] = [(bc_entities_prediction_result['dish_name'][index], best_dish_name) for index, best_dish_name in zip(best_index, best_match_dish_name)]
            return result_dict
        # 处理"推荐"类问题
        # 推荐菜名的话，从菜系、口味、食材方面出发
        if tc_textclassify_result == 'to recommend':
            # 没有任何实体，仅是推荐
            if all(not value for value in bc_entities_prediction_result.values()):
                print("1")
                result_dict['question_type'] = 'just for recommendation'
                return result_dict
            # 从菜系开始
            if bc_entities_prediction_result['cuisine']:
                # 某某菜系中，口味为某某的菜，值得推荐的是什么菜
                if bc_entities_prediction_result['feature']:
                    question_type = 'recommend according to cuisine and feature'
                    result_dict['cuisine'] = bc_entities_prediction_result['cuisine']
                    result_dict['feature'] = bc_entities_prediction_result['feature']
                    result_dict['question_type'] = question_type
                    return result_dict
                # 买了某某食材,想吃某某菜系的菜，值得推荐的是什么菜
                if bc_entities_prediction_result['ingredient']:
                    question_type = 'recommend according to cuisine and ingredient'
                    result_dict['cuisine'] = bc_entities_prediction_result['cuisine']
                    result_dict['ingredient'] = bc_entities_prediction_result['ingredient']
                    result_dict['question_type'] = question_type
                    return result_dict
                # 只有菜系
                question_type = 'recommend according to cuisine'
                result_dict['cuisine'] = bc_entities_prediction_result['cuisine']
                result_dict['question_type'] = question_type
                return result_dict
            # 再口味开始
            if bc_entities_prediction_result['feature']:
                # 买了某某食材,想吃某某口味的菜，值得推荐的是什么菜
                if bc_entities_prediction_result['ingredient']:
                    question_type = 'recommend according to feature and ingredient'
                    result_dict['feature'] = bc_entities_prediction_result['feature']
                    result_dict['ingredient'] = bc_entities_prediction_result['ingredient']
                    result_dict['question_type'] = question_type
                    return result_dict
                # 推荐某某口味的菜
                question_type = 'recommend according to feature'
                result_dict['feature'] = bc_entities_prediction_result['feature']
                result_dict['question_type'] = question_type
                return result_dict
            # 食材
            if bc_entities_prediction_result['ingredient']:
                question_type = 'recommend according to ingredient'
                result_dict['ingredient'] = bc_entities_prediction_result['ingredient']
                result_dict['question_type'] = question_type
                return result_dict
        # 处理"查询菜系"类问题
        # 只能从菜名去找菜系
        if tc_textclassify_result == 'to find cuisine':
            # 同样有菜名不存在或者写错的情况
            question_type = "to find cuisine according to dish_name"
            result_dict['dish_name'] = []
            result_dict['question_type'] = question_type
            # 提取不存在或者写错最相似的菜名
            un_index, un_dish_name, best_index, best_match_dish_name = [], [], [], []
            if correction_dish_name["un_dish_name"]:
                for temple in correction_dish_name['un_dish_name']:
                    un_index.append(temple[0])
                    un_dish_name.append(temple[1])
            if correction_dish_name['best_match_dish_name']:
                for temple in correction_dish_name['best_match_dish_name']:
                    best_index.append(temple[0])
                    best_match_dish_name.append(temple[1])
            # 对bilstm_crf提取的菜名进行分类
            for dish_name in bc_entities_prediction_result['dish_name']:
                index = bc_entities_prediction_result['dish_name'].index(dish_name)
                if (index not in un_index) and (index not in best_index):
                    result_dict['dish_name'].append(bc_entities_prediction_result['dish_name'][index])
            result_dict['un_dish_name'] = un_dish_name
            result_dict['best_match_dish_name'] = [(bc_entities_prediction_result['dish_name'][index], best_dish_name) for index, best_dish_name in zip(best_index, best_match_dish_name)]
            return result_dict
        # 处理"查找特点"类问题
        # 只能从菜名去找特点
        if tc_textclassify_result == 'to find feature':
            # 同样有菜名不存在或者写错的情况
            question_type = "to find feature according to dish_name"
            result_dict['dish_name'] = []
            result_dict['question_type'] = question_type
            # 提取不存在或者写错最相似的菜名
            un_index, un_dish_name, best_index, best_match_dish_name = [], [], [], []
            if correction_dish_name["un_dish_name"]:
                for temple in correction_dish_name['un_dish_name']:
                    un_index.append(temple[0])
                    un_dish_name.append(temple[1])
            if correction_dish_name['best_match_dish_name']:
                for temple in correction_dish_name['best_match_dish_name']:
                    best_index.append(temple[0])
                    best_match_dish_name.append(temple[1])
            # 对bilstm_crf提取的菜名进行分类
            for dish_name in bc_entities_prediction_result['dish_name']:
                index = bc_entities_prediction_result['dish_name'].index(dish_name)
                if (index not in un_index) and (index not in best_index):
                    result_dict['dish_name'].append(bc_entities_prediction_result['dish_name'][index])
            result_dict['un_dish_name'] = un_dish_name
            result_dict['best_match_dish_name'] = [(bc_entities_prediction_result['dish_name'][index], best_dish_name) for index, best_dish_name in zip(best_index, best_match_dish_name)]
            return result_dict
        # 处理"查找食材"类问题
        # 只能从菜名去找食材
        if tc_textclassify_result == 'to find ingredient':
            # 同样有菜名不存在或者写错的情况
            question_type = "to find ingredient according to dish_name"
            result_dict['dish_name'] = []
            result_dict['question_type'] = question_type
            # 提取不存在或者写错最相似的菜名
            un_index, un_dish_name, best_index, best_match_dish_name = [], [], [], []
            if correction_dish_name["un_dish_name"]:
                for temple in correction_dish_name['un_dish_name']:
                    un_index.append(temple[0])
                    un_dish_name.append(temple[1])
            if correction_dish_name['best_match_dish_name']:
                for temple in correction_dish_name['best_match_dish_name']:
                    best_index.append(temple[0])
                    best_match_dish_name.append(temple[1])
            # 对bilstm_crf提取的菜名进行分类
            for dish_name in bc_entities_prediction_result['dish_name']:
                index = bc_entities_prediction_result['dish_name'].index(dish_name)
                if (index not in un_index) and (index not in best_index):
                    result_dict['dish_name'].append(bc_entities_prediction_result['dish_name'][index])
            result_dict['un_dish_name'] = un_dish_name
            result_dict['best_match_dish_name'] = [(bc_entities_prediction_result['dish_name'][index], best_dish_name) for index, best_dish_name in zip(best_index, best_match_dish_name)]
            return result_dict

    def bc_extract_entities(self, question):
        """
        使用BiLSTM-CRF模型提取问题中的实体
        参数:
            question: 输入的问题文本
        返回:
            包含提取实体的字典
        """
        entity_dict = {}
        current_entity = []
        entities = {
            'cook': [],
            'cuisine': [],
            'feature': [],
            'ingredient': []
        }
        entity_types = {
            'cook': 'dish_name',
            'cuisine': 'cuisine',
            'feature': 'feature',
            'ingredient': 'ingredient'
        }
        # 获取预测结果
        _, result = self.BilstmCrfPredictor.model(self.BilstmCrfPredictor.prepare_sequence(question))
        tags = [self.BilstmCrf_id_to_tag[tag] for tag in result]

        # 提取实体
        for index, tag in enumerate(tags):
            if tag == 'O':
                current_entity = []
                continue
            # 检查所有可能的实体类型
            for entity in entities:
                if tag[2:] == entity:
                    if tag.startswith("B-"):
                        current_entity = [index]
                        entities[entity].append(current_entity)
                    else:
                        current_entity.append(index)
        # 构建结果字典
        for entity_type, indices_list in entities.items():
            entity_dict[entity_types[entity_type]] = []
            for indices in indices_list:
                entity_text = ''.join(question[i] for i in indices)
                entity_dict[entity_types[entity_type]].append(entity_text)

        return entity_dict

    def similar(self, a, b):
        """
        计算两个字符串的相似度
        参数:
            a, b: 要比较的两个字符串
        返回:
            相似度分数(0-1之间)
        """
        return SequenceMatcher(None, a, b).ratio()

    def correction_dish_name(self, entity_dict):
        """
        对识别出的菜品名称进行校验和修正
        参数:
            entity_dict: 包含识别实体的字典
        返回:
            包含修正信息的字典
        """
        dict = {
            'un_dish_name': [],
            'best_match_dish_name': []
        }
        cook_list = [item for key, values in entity_dict.items() if key == 'dish_name' for item in values]
        if cook_list:
            for index, cook in enumerate(cook_list):
                # 检查实体是否在region_words中
                if cook not in self.dish_name_words:
                    # 寻找最相似的候选词
                    best_match = None
                    highest_sim = 0
                    for word in self.dish_name_words:
                        sim = self.similar(cook, word)
                        if sim > highest_sim:  # 只保留最高相似度
                            highest_sim = sim
                            best_match = word
                    # 只有当相似度超过阈值才认为有效
                    if best_match and highest_sim > 0.7:  # 可调整阈值
                        dict['best_match_dish_name'].append((index, best_match))
                    else:
                        dict['un_dish_name'].append((index, cook))
        return dict

if __name__ == '__main__':
    qc = QuestionClassifier()
    result = qc.classify('有什么湘菜可以推荐吗？')
    print(result)