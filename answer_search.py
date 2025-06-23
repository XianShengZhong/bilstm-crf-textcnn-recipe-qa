from py2neo import Graph

class AnswerSearcher:
    def __init__(self):
        self.g = Graph('http://localhost:7474', auth=('neo4j', '1234'))

    def search_main(self, result_dic):
        sqls = result_dic['sql']
        answers = []
        # 如果有sql语句
        if sqls:
            for sql in sqls:
                ress = self.g.run(sql).data()
                answers += ress
        final_answers = self.answer_prettify(result_dic, answers)
        return final_answers


    def answer_prettify(self, result_dict, answers):
        final_answer = ''
        if result_dict['question_type'] == 'according to dish_name how to do it':
            # 如果answers为空，就是没有存在的菜名以及写错了菜名
            if not answers:
                # 进一步判断
                # 菜谱中没有的菜名
                un_dish_name_answer = ''
                best_match_dish_name_answer = ''
                if result_dict['un_dish_name']:
                    un_dish_names = '、' .join([un_dish_name for un_dish_name in result_dict['un_dish_name']])
                    un_dish_name_answer = f"布布感到很抱歉（ ＴДＴ）~~，布布不会做{un_dish_names}。\n"
                if result_dict['best_match_dish_name']:
                    best_match_dish_names = '，还有'.join([f'找不到"{erro_dish_name}"但是布布会做"{best_match_dish_name}"' for erro_dish_name, best_match_dish_name in result_dict['best_match_dish_name']])
                    best_match_dish_name_answer = f"布布发现{best_match_dish_names}。\n"
                final_answer = un_dish_name_answer + best_match_dish_name_answer
            # 如果answers为非空，那就是菜谱中存在的菜名
            else:
                # 注意细节，也许输入中有存在的菜名，同时也有不存在的
                un_dish_name_answer = ''
                best_match_dish_name_answer = ''
                if result_dict['un_dish_name']:
                    un_dish_names = '、'.join([un_dish_name for un_dish_name in result_dict['un_dish_name']])
                    un_dish_name_answer = f"布布感到很抱歉（ ＴДＴ）~~，布布不会做{un_dish_names}。\n"
                if result_dict['best_match_dish_name']:
                    best_match_dish_names = '，还有'.join([f'找不到"{erro_dish_name}"但是布布会做"{best_match_dish_name}"' for erro_dish_name, best_match_dish_name in result_dict['best_match_dish_name']])
                    best_match_dish_name_answer = f"布布发现{best_match_dish_names}。\n"
                # 对正确的菜名进行处理
                right_dish_names_format = '\n'.join([f'菜名:{i["菜名"]}\n食材:{i["食材"]}制作步骤:{i["制作步骤"]}\n' for i in answers])
                right_dish_name_answer = f'作为一二的贴身厨师，制作方法早已牢记于心，接下来我将这些年的心得传授于您૮(˶ᵔ ᵕ ᵔ˶)ა:\n{right_dish_names_format}如果还想知道其他菜的制作过程，请尽管问布布吧'
                final_answer = un_dish_name_answer + best_match_dish_name_answer + right_dish_name_answer

        if result_dict['question_type'] == 'just for recommendation':
            # 返回的answers是菜名
            dish_name = '、'.join([i['菜名'] for i in answers])
            final_answer = f'布布推荐以下菜:\n{dish_name}......'

        if result_dict['question_type'] == 'recommend according to cuisine':
            # 可能有多个菜系
            dict = {}
            for i in answers:
                if i['菜系'] not in dict:
                    dict[i['菜系']] =[i['菜名']]
                else:
                    dict[i['菜系']].append(i['菜名'])
            cuisines = dict.keys()
            formats = '根据您的需求，布布推荐以下这些菜，这些菜一二宝都喜欢吃(⑅˃◡˂⑅):\n'
            for cuisine in cuisines:
                dish_name_format = '、'.join(dict[cuisine])
                cuisine_format = f'{cuisine}:{dish_name_format}\n'
                formats += cuisine_format
            final_answer = formats

        # 一般一个菜系加一个口味来推荐
        if result_dict['question_type'] == 'recommend according to cuisine and feature':
            # 返回的answers是菜名
            dish_name = '、'.join([i['菜名'] for i in answers])
            final_answer = f'布布根据您要的菜系以及口味，推荐以下菜，觉得可能适合您的口味(꒪ˊ꒳ˋ꒪):\n{dish_name}......'

        if result_dict['question_type'] == 'recommend according to cuisine and ingredient':
            # 返回的answers是菜名
            dish_name = '、'.join([i['菜名'] for i in answers])
            final_answer = f'布布根据您要的菜系以及食材，推荐以下菜，觉得可能适合您的要求(꒪ˊ꒳ˋ꒪):\n{dish_name}......'

        if result_dict['question_type'] == 'recommend according to feature and ingredient':
            # 返回的answers是菜名
            dish_name = '、'.join([i['菜名'] for i in answers])
            final_answer = f'布布根据您的口味以及食材，推荐以下菜，觉得可能适合您的要求(꒪ˊ꒳ˋ꒪):\n{dish_name}......'

        if result_dict['question_type'] == 'recommend according to feature':
            # 返回的answers是菜名
            dish_name = '、'.join([i['菜名'] for i in answers])
            final_answer = f'布布根据您的口味，推荐以下菜，觉得可能适合您的要求(꒪ˊ꒳ˋ꒪):\n{dish_name}......'

        if result_dict['question_type'] == 'recommend according to ingredient':
            # 返回的answers是菜名
            dish_name = '、'.join([i['菜名'] for i in answers])
            final_answer = f'布布根据您的食材，推荐以下菜，觉得可能适合您的要求(꒪ˊ꒳ˋ꒪):\n{dish_name}......'

        if result_dict['question_type'] == 'to find cuisine according to dish_name':
            # 注意细节，也许输入中有存在的菜名，同时也有不存在的
            un_dish_name_answer = ''
            best_match_dish_name_answer = ''
            if result_dict['un_dish_name']:
                un_dish_names = '、'.join([un_dish_name for un_dish_name in result_dict['un_dish_name']])
                un_dish_name_answer = f"布布感到很抱歉（ ＴДＴ）~~，布布没接触过{un_dish_names}。\n"
            if result_dict['best_match_dish_name']:
                best_match_dish_names = '，还有'.join([f'找不到"{erro_dish_name}"但是布布猜您是想要问"{best_match_dish_name}嘛？"' for erro_dish_name, best_match_dish_name in result_dict['best_match_dish_name']])
                best_match_dish_name_answer = f"布布发现{best_match_dish_names}。\n"
            # 返回的answers是菜系
            cuisine_format = ',\n'.join([f'"{dish_name}"属于"{i["菜系"]}"' for dish_name, i in zip(result_dict['dish_name'], answers)])
            final_answer = un_dish_name_answer + best_match_dish_name_answer + '布布根据您提供的菜名找出了它所属的菜系(〃＾▽＾〃):\n' +cuisine_format

        if result_dict['question_type'] == 'to find feature according to dish_name':
            # 注意细节，也许输入中有存在的菜名，同时也有不存在的
            un_dish_name_answer = ''
            best_match_dish_name_answer = ''
            if result_dict['un_dish_name']:
                un_dish_names = '、'.join([un_dish_name for un_dish_name in result_dict['un_dish_name']])
                un_dish_name_answer = f"布布感到很抱歉（ ＴДＴ）~~，布布没接触过{un_dish_names}。\n"
            if result_dict['best_match_dish_name']:
                best_match_dish_names = '，还有'.join([f'找不到"{erro_dish_name}"但是布布猜您是想要问"{best_match_dish_name}嘛？"' for erro_dish_name, best_match_dish_name in result_dict['best_match_dish_name']])
                best_match_dish_name_answer = f"布布发现{best_match_dish_names}。\n"
            # 返回的answers是口味
            feature_format = ',\n'.join([f'“{dish_name}”的口味为:{i["口味"]}' for dish_name, i in zip(result_dict['dish_name'], answers)])
            final_answer = un_dish_name_answer + best_match_dish_name_answer + '布布根据您提供的菜名找出了它的口味(〃＾▽＾〃):\n' + feature_format

        if result_dict['question_type'] == 'to find ingredient according to dish_name':
            # 注意细节，也许输入中有存在的菜名，同时也有不存在的
            un_dish_name_answer = ''
            best_match_dish_name_answer = ''
            if result_dict['un_dish_name']:
                un_dish_names = '、'.join([un_dish_name for un_dish_name in result_dict['un_dish_name']])
                un_dish_name_answer = f"布布感到很抱歉（ ＴДＴ）~~，布布没接触过{un_dish_names}。\n"
            if result_dict['best_match_dish_name']:
                best_match_dish_names = '，还有'.join([f'找不到"{erro_dish_name}"但是布布猜您是想要问"{best_match_dish_name}嘛？"' for erro_dish_name, best_match_dish_name in result_dict['best_match_dish_name']])
                best_match_dish_name_answer = f"布布发现{best_match_dish_names}。\n"
            # 返回的answers是口味
            ingredient_format = ',\n'.join([f'“{dish_name}”所需食材为:{i["食材"]}' for dish_name, i in zip(result_dict['dish_name'], answers)])
            final_answer = un_dish_name_answer + best_match_dish_name_answer + '布布根据您提供的菜名找出了它所需的食材(〃＾▽＾〃):\n' + ingredient_format

        return final_answer


if __name__ == '__main__':
    a = AnswerSearcher()
    result = a.search_main({'dish_name': [], 'question_type': 'according dish_name how to do', 'un_dish_name': ['宫保鸡丁'], 'best_match_dish_name': [('辣椒炒屎', '辣椒炒肉'),('土豆死', '土豆丝')], 'sql': []})
    print(result)