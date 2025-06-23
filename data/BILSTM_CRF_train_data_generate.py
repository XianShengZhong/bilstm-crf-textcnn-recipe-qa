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
        读取多词格式的文本文件（用顿号分隔），返回去重后的单词列表
        每行多个词用顿号分隔，自动拆分并去除空白字符
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

def single_convert_to_bio_format(word_list, label=None):
    """
        将单个词语转换为BIO标注格式（单字词只标B，多字词标B-I-E）
    """
    result = []
    for word in word_list:
        if len(word) == 0:
            continue
        # 处理第一个字
        result.append(f"{word[0]} B-{label}")
        # 处理中间的字（如果有）
        for char in word[1:-1]:
            result.append(f"{char} I-{label}")
        # 处理最后一个字（如果长度>1）
        if len(word) > 1:
            result.append(f"{word[-1]} E-{label}")
    return '\n'.join(result) + '\n'

def double_convert_to_bio_format(word_list, label=None):
    """
        将多个词语（用顿号分隔）转换为BIO标注格式，顿号标记为O
    """
    result = []
    for i, word in enumerate(word_list):
        if len(word) == 0:
            continue
        # 处理第一个字
        result.append(f"{word[0]} B-{label}")
        # 处理中间的字（如果有）
        for char in word[1:-1]:
            result.append(f"{char} I-{label}")
        # 处理最后一个字（如果长度>1）
        if len(word) > 1:
            result.append(f"{word[-1]} E-{label}")
        # 如果不是最后一个词，添加顿号和O标签
        if i != len(word_list) - 1:
            result.append("、 O")
    return '\n'.join(result) + '\n'

def convert_to_bio_format(word_list, label=None):
    """
        自动选择单字或多字转换方式
    """
    if len(word_list) == 1:
        return single_convert_to_bio_format(word_list,label)
    elif len(word_list) >= 2:
        return double_convert_to_bio_format(word_list,label)
    else:
        return None

# 生成单句数据
def generate_sentence(dish=None):
    """
        生成带有BIO标注的随机菜谱问答句子
    """
    # 如果没有指定菜品，则随机选择一个
    if dish is None:
        dish = [random.choice(dish_names)]
    else:
        dish = [dish]
    # 随机选择一个菜系
    cuisine = [random.choice(cuisine_types)]
    # 随机选择菜品特征（1-3个）
    num_features = random.randint(1, 2)
    selected_features = random.sample(features_list, num_features)
    # 随机选择原料（1-4个）
    num_ingredients = random.randint(1, 2)
    selected_ingredients = random.sample(ingredients_list, num_ingredients)

    dish = convert_to_bio_format(dish, label='cook')
    cuisine = convert_to_bio_format(cuisine, label='cuisine')
    feature = convert_to_bio_format(selected_features, label='feature')
    ingredient = convert_to_bio_format(selected_ingredients, label='ingredient')
    sentences = [f'我 O\n买 O\n了 O\n{ingredient}等 O\n材 O\n料 O\n， O\n同 O\n时 O\n我 O\n想 O\n吃 O\n{feature}的 O\n{cuisine}的 O\n菜 O\n， O\n可 O\n以 O\n做 O\n什 O\n么 O\n菜 O\n， O\n或 O\n者 O\n可 O\n以 O\n告 O\n诉 O\n我 O\n{dish}怎 O\n做 O\n。 O\n  O\n',
                 f'请 O\n问 O\n{dish}怎 O\n么 O\n做 O\n， O\n它 O\n是 O\n属 O\n于 O\n{cuisine}嘛 O\n？ O\n我 O\n想 O\n用 O\n{ingredient}做 O\n{feature}的 O\n菜 O\n， O\n请 O\n问 O\n怎 O\n么 O\n做 O\n。 O\n  O\n',
                 f"请 O\n问 O\n{dish}怎 O\n么 O\n做 O\n？ O\n需 O\n要 O\n{ingredient}吗 O\n？ O\n我 O\n想 O\n做 O\n{feature}的 O\n{cuisine}。 O\n  O\n",
                 f"这 O\n道 O\n{cuisine}的 O\n{dish}非 O\n常 O\n有 O\n名 O\n， O\n它 O\n的 O\n特 O\n点 O\n是 O\n{feature}， O\n主 O\n要 O\n用 O\n到 O\n{ingredient}。 O\n  O\n",
                 f"我 O\n想 O\n点 O\n一 O\n道 O\n{feature}的 O\n{cuisine}， O\n最 O\n好 O\n是 O\n用 O\n{ingredient}做 O\n的 O\n{dish}。 O\n  O\n",
                 f"请 O\n教 O\n我 O\n做 O\n{dish}， O\n我 O\n准 O\n备 O\n了 O\n{ingredient}， O\n希 O\n望 O\n做 O\n出 O\n{feature}的 O\n效 O\n果 O\n。 O\n  O\n",
                 f"{cuisine}的 O\n{dish}和 O\n另 O\n一 O\n道 O\n菜 O\n比 O\n较 O\n， O\n哪 O\n个 O\n更 O\n容 O\n易 O\n做 O\n？ O\n需 O\n要 O\n{ingredient}吗 O\n？ O\n  O\n",
                 f"如 O\n果 O\n你 O\n喜 O\n欢 O\n{feature}的 O\n菜 O\n， O\n我 O\n推 O\n荐 O\n{cuisine}的 O\n{dish}， O\n主 O\n要 O\n用 O\n{ingredient}制 O\n作 O\n。 O\n  O\n",
                 f"做 O\n{dish}需 O\n要 O\n准 O\n备 O\n哪 O\n些 O\n材 O\n料 O\n？ O\n是 O\n否 O\n需 O\n要 O\n{ingredient}？ O\n这 O\n道 O\n{cuisine}有 O\n什 O\n么 O\n{feature}？ O\n  O\n",
                 f"请 O\n问 O\n如 O\n何 O\n才 O\n能 O\n把 O\n{dish}做 O\n得 O\n{feature}？ O\n我 O\n有 O\n{ingredient}， O\n这 O\n是 O\n{cuisine}的 O\n传 O\n统 O\n做 O\n法 O\n吗 O\n？ O\n  O\n",
                 f"昨 O\n天 O\n吃 O\n的 O\n{cuisine}的 O\n{dish}真 O\n不 O\n错 O\n， O\n{feature}， O\n特 O\n别 O\n是 O\n用 O\n{ingredient}的 O\n部 O\n分 O\n很 O\n出 O\n彩 O\n。 O\n  O\n",
                 f"我 O\n想 O\n用 O\n{ingredient}创 O\n新 O\n一 O\n道 O\n{cuisine}， O\n做 O\n出 O\n{feature}的 O\n效 O\n果 O\n， O\n能 O\n借 O\n鉴 O\n{dish}的 O\n做 O\n法 O\n吗 O\n？ O\n  O\n",
                 f"请 O\n详 O\n细 O\n讲 O\n解 O\n一 O\n下 O\n{cuisine}的 O\n{dish}的 O\n制 O\n作 O\n步 O\n骤 O\n， O\n特 O\n别 O\n是 O\n如 O\n何 O\n处 O\n理 O\n{ingredient}才 O\n能 O\n达 O\n到 O\n{feature}的 O\n效 O\n果 O\n？ O\n  O\n",
                 f"传 O\n统 O\n的 O\n{cuisine}做 O\n{dish}和 O\n现 O\n代 O\n做 O\n法 O\n有 O\n什 O\n么 O\n区 O\n别 O\n？ O\n哪 O\n种 O\n更 O\n能 O\n体 O\n现 O\n{feature}的 O\n特 O\n点 O\n？ O\n用 O\n{ingredient}时 O\n有 O\n什 O\n么 O\n不 O\n同 O\n要 O\n求 O\n？ O\n  O\n",
                 f"中 O\n秋 O\n节 O\n想 O\n给 O\n家 O\n人 O\n做 O\n一 O\n道 O\n{feature}的 O\n{cuisine}， O\n推 O\n荐 O\n用 O\n{ingredient}制 O\n作 O\n{dish}， O\n请 O\n问 O\n需 O\n要 O\n注 O\n意 O\n哪 O\n些 O\n细 O\n节 O\n才 O\n能 O\n做 O\n得 O\n地 O\n道 O\n？ O\n  O\n",
                 f"作 O\n为 O\n健 O\n康 O\n饮 O\n食 O\n， O\n{cuisine}的 O\n{dish}适 O\n合 O\n经 O\n常 O\n食 O\n用 O\n吗 O\n？ O\n用 O\n{ingredient}制 O\n作 O\n时 O\n如 O\n何 O\n减 O\n少 O\n油 O\n脂 O\n摄 O\n入 O\n又 O\n能 O\n保 O\n持 O\n{feature}的 O\n口 O\n感 O\n？ O\n  O\n",
                 f"做 O\n{feature}的 O\n{cuisine}的 O\n{dish}时 O\n， O\n用 O\n铁 O\n锅 O\n还 O\n是 O\n不 O\n粘 O\n锅 O\n更 O\n合 O\n适 O\n？ O\n处 O\n理 O\n{ingredient}时 O\n需 O\n要 O\n什 O\n么 O\n特 O\n殊 O\n的 O\n厨 O\n具 O\n吗 O\n？ O\n  O\n",
                 f"如 O\n果 O\n只 O\n有 O\n1 O\n小 O\n时 O\n准 O\n备 O\n晚 O\n餐 O\n， O\n能 O\n做 O\n好 O\n{cuisine}的 O\n{dish}吗 O\n？ O\n用 O\n{ingredient}制 O\n作 O\n时 O\n哪 O\n些 O\n步 O\n骤 O\n可 O\n以 O\n提 O\n前 O\n准 O\n备 O\n？ O\n如 O\n何 O\n快 O\n速 O\n实 O\n现 O\n{feature}的 O\n效 O\n果 O\n？ O\n  O\n",
                 f"做 O\n{cuisine}的 O\n{dish}时 O\n如 O\n果 O\n没 O\n有 O\n{ingredient}， O\n可 O\n以 O\n用 O\n什 O\n么 O\n替 O\n代 O\n品 O\n？ O\n替 O\n换 O\n后 O\n还 O\n能 O\n保 O\n持 O\n{feature}的 O\n风 O\n味 O\n吗 O\n？ O\n  O\n",
                 f"请 O\n问 O\n专 O\n业 O\n厨 O\n师 O\n在 O\n制 O\n作 O\n{cuisine}的 O\n{dish}时 O\n有 O\n哪 O\n些 O\n特 O\n别 O\n的 O\n技 O\n巧 O\n？ O\n特 O\n别 O\n是 O\n处 O\n理 O\n{ingredient}和 O\n实 O\n现 O\n{feature}效 O\n果 O\n方 O\n面 O\n。 O\n  O\n",
                 f"{cuisine}的 O\n{dish}有 O\n什 O\n么 O\n有 O\n趣 O\n的 O\n历 O\n史 O\n故 O\n事 O\n？ O\n为 O\n什 O\n么 O\n会 O\n选 O\n用 O\n{ingredient}作 O\n为 O\n主 O\n料 O\n？ O\n{feature}这 O\n个 O\n特 O\n点 O\n是 O\n如 O\n何 O\n形 O\n成 O\n的 O\n？ O\n  O\n",
                 f"在 O\n冬 O\n季 O\n制 O\n作 O\n{cuisine}的 O\n{dish}有 O\n什 O\n么 O\n特 O\n别 O\n注 O\n意 O\n事 O\n项 O\n？ O\n{ingredient}的 O\n选 O\n购 O\n和 O\n处 O\n理 O\n与 O\n其 O\n他 O\n季 O\n节 O\n有 O\n何 O\n不 O\n同 O\n？ O\n如 O\n何 O\n在 O\n寒 O\n冷 O\n天 O\n气 O\n保 O\n持 O\n{feature}的 O\n品 O\n质 O\n？ O\n  O\n",
                 f"招 O\n待 O\n重 O\n要 O\n客 O\n人 O\n时 O\n， O\n如 O\n何 O\n将 O\n{cuisine}的 O\n{dish}摆 O\n盘 O\n得 O\n更 O\n加 O\n精 O\n美 O\n？ O\n搭 O\n配 O\n{ingredient}时 O\n有 O\n哪 O\n些 O\n创 O\n意 O\n做 O\n法 O\n？ O\n如 O\n何 O\n突 O\n出 O\n{feature}这 O\n个 O\n亮 O\n点 O\n？ O\n  O\n",
                 f"我 O\n已 O\n经 O\n会 O\n基 O\n础 O\n的 O\n{dish}做 O\n法 O\n， O\n如 O\n何 O\n进 O\n一 O\n步 O\n提 O\n升 O\n到 O\n专 O\n业 O\n水 O\n平 O\n？ O\n在 O\n{ingredient}的 O\n处 O\n理 O\n和 O\n{feature}的 O\n把 O\n控 O\n上 O\n有 O\n哪 O\n些 O\n高 O\n阶 O\n技 O\n巧 O\n？ O\n  O\n",
                 f"在 O\n{cuisine}的 O\n发 O\n源 O\n地 O\n， O\n人 O\n们 O\n做 O\n{dish}时 O\n有 O\n哪 O\n些 O\n与 O\n众 O\n不 O\n同 O\n的 O\n习 O\n惯 O\n？ O\n为 O\n什 O\n么 O\n当 O\n地 O\n人 O\n特 O\n别 O\n强 O\n调 O\n{ingredient}的 O\n某 O\n种 O\n处 O\n理 O\n方 O\n式 O\n？ O\n{feature}是 O\n如 O\n何 O\n成 O\n为 O\n标 O\n志 O\n性 O\n特 O\n点 O\n的 O\n？ O\n  O\n",
                 f"如 O\n何 O\n在 O\n传 O\n统 O\n的 O\n{cuisine}的 O\n{dish}基 O\n础 O\n上 O\n进 O\n行 O\n创 O\n新 O\n？ O\n保 O\n留 O\n{feature}特 O\n点 O\n的 O\n同 O\n时 O\n， O\n能 O\n用 O\n{ingredient}做 O\n哪 O\n些 O\n现 O\n代 O\n化 O\n改 O\n良 O\n？ O\n  O\n",
                 f"从 O\n食 O\n品 O\n科 O\n学 O\n角 O\n度 O\n， O\n{cuisine}的 O\n{dish}中 O\n{ingredient}的 O\n化 O\n学 O\n反 O\n应 O\n是 O\n怎 O\n样 O\n产 O\n生 O\n{feature}的 O\n？ O\n温 O\n度 O\n和 O\n时 O\n间 O\n如 O\n何 O\n影 O\n响 O\n最 O\n终 O\n效 O\n果 O\n？ O\n  O\n",
                 f"我 O\n奶 O\n奶 O\n做 O\n的 O\n{cuisine}的 O\n{dish}特 O\n别 O\n好 O\n吃 O\n， O\n但 O\n她 O\n总 O\n是 O\n凭 O\n感 O\n觉 O\n放 O\n{ingredient}， O\n如 O\n何 O\n将 O\n这 O\n种 O\n家 O\n庭 O\n做 O\n法 O\n量 O\n化 O\n并 O\n保 O\n留 O\n{feature}的 O\n特 O\n色 O\n？ O\n  O\n",
                 f"高 O\n级 O\n餐 O\n厅 O\n制 O\n作 O\n{cuisine}的 O\n{dish}时 O\n有 O\n哪 O\n些 O\n家 O\n庭 O\n厨 O\n房 O\n很 O\n难 O\n达 O\n到 O\n的 O\n标 O\n准 O\n？ O\n特 O\n别 O\n是 O\n{ingredient}的 O\n处 O\n理 O\n和 O\n{feature}的 O\n呈 O\n现 O\n方 O\n面 O\n。 O\n  O\n",
                 f"我 O\n多 O\n次 O\n尝 O\n试 O\n做 O\n{cuisine}的 O\n{dish}但 O\n总 O\n是 O\n失 O\n败 O\n， O\n特 O\n别 O\n是 O\n{ingredient}的 O\n部 O\n分 O\n和 O\n{feature}的 O\n实 O\n现 O\n， O\n可 O\n能 O\n是 O\n哪 O\n些 O\n关 O\n键 O\n步 O\n骤 O\n出 O\n了 O\n问 O\n题 O\n？ O\n  O\n",
                 f"如 O\n何 O\n将 O\n{cuisine}的 O\n{dish}与 O\n西 O\n式 O\n烹 O\n饪 O\n结 O\n合 O\n？ O\n用 O\n{ingredient}时 O\n可 O\n以 O\n借 O\n鉴 O\n哪 O\n些 O\n国 O\n际 O\n手 O\n法 O\n？ O\n如 O\n何 O\n在 O\n融 O\n合 O\n中 O\n保 O\n留 O\n{feature}的 O\n本 O\n土 O\n特 O\n色 O\n？ O\n  O\n",
                 f"春 O\n节 O\n期 O\n间 O\n制 O\n作 O\n{cuisine}的 O\n{dish}有 O\n什 O\n么 O\n特 O\n别 O\n寓 O\n意 O\n？ O\n如 O\n何 O\n通 O\n过 O\n{ingredient}的 O\n搭 O\n配 O\n和 O\n{feature}的 O\n呈 O\n现 O\n增 O\n添 O\n节 O\n日 O\n气 O\n氛 O\n？ O\n  O\n"
                ]

    return ''.join(random.sample(sentences, 1))


# 生成文件
def generate_file(filename, num_lines):
    """
        生成训练数据文件，确保每个菜品至少出现一次
    """
    with open(filename, 'w', encoding='utf-8') as f:
        # 首先确保每个菜品都被使用至少一次
        for dish in dish_names:
            sentence = generate_sentence(dish)
            f.write(sentence)


# 生成数据，确保每个菜品都被使用
generate_file('BILSTM_CRF_train.txt', len(dish_names))
print(f"文件生成完成，共{len(dish_names)}行数据。每个菜品至少出现一次。")