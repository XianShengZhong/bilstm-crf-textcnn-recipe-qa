class QuestionPaser:
    """
       问题解析器
       功能：
           1. 根据问题分类结果构建Cypher查询语句
           2. 处理不同问题类型对应的查询逻辑
           3. 对口味特征词进行标准化处理
    """
    def __init__(self):
        # 加载口味特征词
        # 可能输入的特点太过专一，从而可能查询不到
        self.feature_words = ['酸', '甜', '苦', '辣', '咸', '鲜', '香 ', '醇', '嫩', '脆', '酥', '麻','软', '滑', '烂', '胃', '色', '艳',
                          '亮', '美', '爽', '诱', '宜', '养', '济', '酱', '肥', '化', '黄', '红', '亮', '棕', '白', '绿', '炸', '蒸',
                          '炒', '烧', '煨', '焖', '烤', '煎', '煨', '烩', '拌']

    def parser_main(self, result_dict):
        """
        主解析方法
        参数:
            result_dict: 包含问题分类结果和实体信息的字典
        返回:
            添加了Cypher查询语句的结果字典
        """
        sql = []
        # 对result_dict['question_type']进行分类解析
        # 处理"根据菜名查询做法"类问题
        if result_dict['question_type'] == 'according to dish_name how to do it':
            # 存在正确的菜名
            if result_dict['dish_name']:
                # 可能有多个菜
                dish_list = ', '.join([f'"{dish}"' for dish in result_dict['dish_name']])
                sql = [f'MATCH (p:Dish_Name) WHERE p.name IN [{dish_list}]RETURN p.name AS 菜名, p.ingredients AS 食材,p.steps AS 制作步骤']
                result_dict['sql'] = sql
            else:
                # 不存在正确的菜名，就没有sql语句
                result_dict['sql'] = sql

        # 处理”没条件推荐“类问题
        if result_dict['question_type'] == 'just for recommendation':
            sql = ['MATCH (p:Dish_Name) RETURN p.name AS 菜名 ORDER BY rand() LIMIT 5']
            result_dict['sql'] = sql

        # 处理"根据菜系推荐菜品"类问题
        if result_dict['question_type'] == 'recommend according to cuisine':
            # 可能存在多个菜系
            sql = ['MATCH (q:Cuisines)<--(p:Dish_Name) WHERE q.name = "{0}" RETURN p.name AS 菜名 ,q.name AS 菜系 ORDER BY rand() LIMIT 5'.format(i) for i in result_dict['cuisine']]
            result_dict['sql'] = sql

        # 处理"根据菜系和口味推荐菜品"类问题
        if result_dict['question_type'] == 'recommend according to cuisine and feature':
            # 一般一个问句只有一个菜系和多个口味
            cuisine = result_dict['cuisine'][0]  # 假设只有一个菜系
            features = result_dict['feature']  # 可能有多个专业的口味特征描述，需要简化
            singel_features = []
            for feature in features:
                for singel_feature in self.feature_words:
                    if singel_feature in feature:
                        singel_features.append(singel_feature)
            conditions = ' AND '.join([f'p.features CONTAINS "{k}"' for k in singel_features])
            sql = [f'MATCH (q:Cuisines)<--(p:Dish_Name) WHERE q.name = "{cuisine}" AND {conditions} WITH p, rand() AS random ORDER BY random LIMIT 5 RETURN p.name AS 菜名']
            result_dict['sql'] = sql

        # 处理"根据菜系和食材推荐菜品"类问题
        if result_dict['question_type'] == 'recommend according to cuisine and ingredient':
            # 一般一个问句只有一个菜系和多种食材
            cuisine = result_dict['cuisine'][0]  # 假设只有一个菜系
            conditions = ' AND '.join([f'p.ingredients CONTAINS "{k}"' for k in result_dict['ingredient']])
            sql = [f'MATCH (q:Cuisines)<--(p:Dish_Name) WHERE q.name = "{cuisine}" AND {conditions} WITH p, rand() AS random ORDER BY random LIMIT 5 RETURN p.name AS 菜名']
            result_dict['sql'] = sql

        # 处理"根据口味和食材推荐菜品"类问题
        if result_dict['question_type'] == 'recommend according to feature and ingredient':
            # 可能有多个食材和多个口味
            features = result_dict['feature']  # 可能有多个专业的口味特征描述，需要简化
            singel_features = []
            for feature in features:
                for singel_feature in self.feature_words:
                    if singel_feature in feature:
                        singel_features.append(singel_feature)
            conditions1 = ' AND '.join([f'p.ingredients CONTAINS "{k}"' for k in result_dict['ingredient']])
            conditions2 = 'AND'.join([f' p.features CONTAINS "{singel_feature}"' for singel_feature in singel_features])
            sql = [f'MATCH (p:Dish_Name) WHERE {conditions2} AND {conditions1} WITH p, rand() AS random ORDER BY random LIMIT 5 RETURN p.name AS 菜名']
            result_dict['sql'] = sql

        # 处理"根据口味推荐菜品"类问题
        if result_dict['question_type'] == 'recommend according to feature':
            # 可能有多个口味
            features = result_dict['feature']  # 可能有多个专业的口味特征描述，需要简化
            singel_features = []
            for feature in features:
                for singel_feature in self.feature_words:
                    if singel_feature in feature:
                        singel_features.append(singel_feature)
            conditions = ' AND '.join([f'p.features CONTAINS "{k}"' for k in singel_features])
            sql = [f'MATCH (p:Dish_Name) WHERE {conditions} WITH p, rand() AS random ORDER BY random LIMIT 5 RETURN p.name AS 菜名']
            result_dict['sql'] = sql

        # 处理"根据食材推荐菜品"类问题
        if result_dict['question_type'] == 'recommend according to ingredient':
            # 可能有多种食材
            conditions = ' AND '.join([f'p.ingredients CONTAINS "{k}"' for k in result_dict['ingredient']])
            sql = [f'MATCH (p:Dish_Name) WHERE {conditions} WITH p, rand() AS random ORDER BY random LIMIT 5 RETURN p.name AS 菜名']
            result_dict['sql'] = sql

        # 处理"根据菜名查询菜系"类问题
        if result_dict['question_type'] == 'to find cuisine according to dish_name':
            # 存在正确的菜名
            if result_dict['dish_name']:
                # 可能有多个菜
                dish_list = ', '.join([f'"{dish}"' for dish in result_dict['dish_name']])
                sql = [f'MATCH (q:Cuisines)<--(p:Dish_Name) WHERE p.name IN [{dish_list}]RETURN q.name AS 菜系']
                result_dict['sql'] = sql
            else:
                # 不存在正确的菜名，就没有sql语句
                result_dict['sql'] = sql

        # 处理"根据菜名查询特点"类问题
        if result_dict['question_type'] == 'to find feature according to dish_name':
            # 存在正确的菜名
            if result_dict['dish_name']:
                # 可能有多个口味
                dish_list = ', '.join([f'"{dish}"' for dish in result_dict['dish_name']])
                sql = [f'MATCH (p:Dish_Name) WHERE p.name IN [{dish_list}] RETURN p.features AS 口味']
                result_dict['sql'] = sql
            else:
                # 不存在正确的菜名，就没有sql语句
                result_dict['sql'] = sql

        # 处理"根据菜名查询食材"类问题
        if result_dict['question_type'] == 'to find ingredient according to dish_name':
            # 存在正确的菜名
            if result_dict['dish_name']:
                # 可能有多个口味
                dish_list = ', '.join([f'"{dish}"' for dish in result_dict['dish_name']])
                sql = [f'MATCH (p:Dish_Name) WHERE p.name IN [{dish_list}] RETURN p.ingredients AS 食材']
                result_dict['sql'] = sql
            else:
                # 不存在正确的菜名，就没有sql语句
                result_dict['sql'] = sql

        return  result_dict

if __name__ == '__main__':
    qp = QuestionPaser()
    result = qp.parser_main({'dish_name': ['干烧鱼翅'], 'question_type': 'according dish_name how to do', 'un_dish_name': [], 'best_match_dish_name': []})
    print(result)   # {'dish_name': ['干烧鱼翅'], 'question_type': 'according dish_name how to do', 'un_dish_name': [], 'best_match_dish_name': [], 'sql': ['MATCH (p:Dish_Name) WHERE p.name IN ["干烧鱼翅"]RETURN p.name AS 菜名, p.ingredients AS 食材,p.steps AS 制作步骤']}
