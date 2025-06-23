import random


def read_line_single(filename):
    """
    读取单行格式的文本文件，返回去重后的单词列表
    每行一个词，自动去除首尾空白字符
    """
    word_list = []
    with open(filename, 'r', encoding='utf-8') as f:  # 使用 'r' 模式读取
        for word in f:
            word = word.strip()  # 去除首尾空白字符（包括换行符）
            if word and word not in word_list:  # 确保非空且不重复
                word_list.append(word)
    return word_list
def read_line_double(filename):
    """
    读取多词格式的文本文件（用顿号分隔）
    返回去重后的单词列表
    """
    word_list = []
    with open(filename, 'r', encoding='utf-8') as f:  # 使用 'r' 模式读取
        for line in f:
            line = line.strip().split('、')  # 去除首尾空白字符（包括换行符）
            for word in line:
                if word and word not in word_list:  # 确保非空且不重复
                    word_list.append(word)
    return word_list

# 菜品名称、菜系、特征、原料的示例数据
dish_names = read_line_single('dish_name.txt')
cuisine_types = read_line_single('cuisines.txt')
features_list = read_line_double('feature.txt')
ingredients_list = read_line_double('ingredient.txt')

def single_convert_to_bio_format(word_list):
    """
    处理单个词语（直接返回）
    """
    result = word_list[0]
    return result

def double_convert_to_bio_format(word_list):
    """
    处理多个词语（用顿号连接）
    """
    result = []
    for i, word in enumerate(word_list):
        if len(word) == 0:
            continue
        # 处理第一个字
        result.append(word)
        if i != len(word_list) - 1:
            result.append("、")
    return ''.join(result)

def convert_to_bio_format(word_list):
    """
    自动选择单字或多字转换方式
    """
    if len(word_list) == 1:
        return single_convert_to_bio_format(word_list)
    elif len(word_list) >= 2:
        return double_convert_to_bio_format(word_list)
    else:
        return None

# 生成单句数据
def generate_sentence(dish=None):
    """
    生成随机菜谱问答句子
    返回格式：标签 + 句子
    """
    # 随机选择菜名（1-2个）
    num_features = random.randint(1, 2)
    selected_dish = random.sample(dish_names, num_features)
    # 随机选择一个菜系
    cuisine = [random.choice(cuisine_types)]
    # 随机选择菜品特征（1-3个）
    num_features = random.randint(1, 2)
    selected_features = random.sample(features_list, num_features)
    # 随机选择原料（1-4个）
    num_ingredients = random.randint(1, 2)
    selected_ingredients = random.sample(ingredients_list, num_ingredients)

    dish = convert_to_bio_format(selected_dish)
    cuisine = convert_to_bio_format(cuisine)
    feature = convert_to_bio_format(selected_features)
    ingredient = convert_to_bio_format(selected_ingredients)
    sentences = [
                # to do
                 f"请问{dish}怎么做？需要准备{ingredient}吗？",
                 f"我想学做{cuisine}的{dish}，听说要用到{ingredient}，具体步骤是什么？",
                 f"如何用{ingredient}做出{feature}的{dish}？",
                 f"请教我做{cuisine}的{dish}，我已经买了{ingredient}。",
                 f"{dish}的传统做法是什么？需要哪些材料？",
                 f"做{feature}的{dish}有什么技巧？要用到{ingredient}吗？",
                 f"请问如何制作{cuisine}的{dish}？特别是{feature}的部分怎么处理？",
                 f"我想用{ingredient}做{dish}，求详细步骤。",
                 f"怎样才能把{dish}做得{feature}？需要准备什么材料？",
                 f"专业厨师是怎么做{cuisine}的{dish}的？",
                # to recommend    10
                 f"有什么{cuisine}可以推荐吗？最好是用{ingredient}做的。",
                 f"我想吃{feature}的菜，有什么推荐吗？",
                 f"用{ingredient}可以做什么好吃的{cuisine}？",
                 f"能推荐几道{feature}的{cuisine}吗？",
                 f"有什么用{ingredient}做的经典{cuisine}推荐？",
                 f"我想尝试{cuisine}，有什么{feature}的菜品推荐？",
                 f"能推荐一些适合用{ingredient}做的菜吗？",
                 f"有什么简单易做的{cuisine}推荐吗？最好能有{feature}的特点。",
                 f"请推荐几道{cuisine}的特色菜，最好包含{ingredient}。",
                 f"有什么{feature}的{cuisine}适合家庭聚餐？"
                # to find cuisine  20
                 f"请问{dish}属于什么菜系？",
                 f"{dish}是哪个地方的特色菜？",
                 f"能告诉我{dish}属于哪种菜系吗？",
                 f"我想知道{feature}的{dish}是哪个菜系的？",
                 f"用{ingredient}做的{dish}通常属于什么菜系？",
                 f"哪种菜系会用到{ingredient}来做{dish}？",
                 f"{dish}的传统做法属于什么菜系？",
                 f"有什么菜系擅长做{feature}的菜？",
                 f"哪些菜系会用到{ingredient}作为主要食材？",
                 f"请问{feature}口味的菜通常属于什么菜系？"
                # to find feature   30
                 f"{dish}通常是什么口味的？",
                 f"请问{dish}的主要风味特点是什么？",
                 f"如何描述{cuisine}的{dish}的口感？",
                 f"用{ingredient}做的菜一般有什么特点？",
                 f"{cuisine}的菜品通常有什么特点？",
                 f"请问{dish}最突出的风味是什么？",
                # to find ingredient   36
                 f"做{dish}一般需要哪些食材？",
                 f"请问{cuisine}的{dish}主要用什么原料？",
                 f"传统{cuisine}的{dish}必须用哪些材料？",
                 f"哪些食材搭配能让{dish}更{feature}？",
                 f"做{feature}的{dish}需要特别注意哪些材料？",
                 f"请问{dish}的核心食材是什么？",
                 f"哪些食材适合用来做{cuisine}？",
                 f"正宗{dish}必须包含哪些原料？"
                ]
    sentences_tags = ['to do', 'to recommend','to find cuisine', 'to find feature', 'to find ingredient']

    # 同时获取索引和句子
    random_index, selected_sentence = random.choice(list(enumerate(sentences)))
    tag =None
    if random_index <= 9:
        tag = sentences_tags[0]
    elif (random_index > 9) and (random_index <= 19):
        tag = sentences_tags[1]
    elif (random_index > 19) and (random_index <= 29):
        tag = sentences_tags[2]
    elif (random_index > 29) and (random_index <= 35):
        tag = sentences_tags[3]
    elif (random_index > 35) and (random_index <= 43):
        tag = sentences_tags[4]
    return tag + "    " + selected_sentence + '\n'


# 生成文件
def generate_file(filename, num_lines):
    """
    生成训练数据文件
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for num in range(num_lines):
            sentence = generate_sentence()
            f.write(sentence)

# 生成数据，确保每个菜品都被使用
generate_file('TEXTCNN_train.txt', 10000)
print(f"文件生成完成，共10000行数据。")