import re
import json

# 读取并清洗原始文本
with open("cookbook_unhandled.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    cleaned_text = re.sub(r'\s*第\d+章\s*<br /><br /><br /><br />', '', text)
    cleaned_text = re.sub(r'<br />\s*', '', cleaned_text)  # 修正正则表达式

# 匹配所有菜谱条目
# 匹配格式：【菜名】...【所属菜系】...【特点】...【原料】...【制作过程】...
pattern = r'【菜名】\s*(.*?)\s*【所属菜系】\s*(.*?)\s*【特点】\s*(.*?)\s*【原料】\s*(.*?)\s*【制作过程】\s*(.*?)(?=\s*【菜名】|$)'
recipes = re.findall(pattern, cleaned_text, re.DOTALL)

# 将所有字典保存为 JSON Lines 格式（每行一个字典）
all_dish_name = []
with open("data.json", "w", encoding="utf-8") as f:
    for recipe in recipes:
        name, cuisine, features, ingredients, steps = recipe
        if name not in all_dish_name:
            all_dish_name.append(name)
            # 构建菜谱字典
            dish_dict = {
                "dish_name": name.strip(),
                "cuisine": cuisine.strip(),
                "features": features.strip(),
                "ingredients": ingredients.strip(),
                "steps": re.sub(r"。([^。]*)$",'',steps.strip())
            }
            # 写入JSON文件（每行一个字典）
            json.dump(dish_dict, f, ensure_ascii=False)
            f.write("\n")  # 每行一个字典
        else:
            continue # 跳过重复菜名