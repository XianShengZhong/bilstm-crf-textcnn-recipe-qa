import os
import json
from py2neo import Graph, Node
from tqdm import tqdm  # 导入进度条库

"""
    CookGraph类
    获取节点以及节点关系
    创建图谱
"""
class CookGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data/data.json')
        self.g = Graph("http://localhost:7474", auth=("neo4j", "1234"))

    """
        获取数据集中的菜名节点、菜系节点，以及关系
    """
    def read_nodes(self):
        # 存放菜名
        dish_names = []
        # 存放菜系种类
        cuisines = []
        # 存放菜名于菜系之间的关系
        rels_cuisine = []
        # 存放菜肴的信息
        cook_infos = []

        # 使用tqdm显示读取进度
        with open(self.data_path, encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)  # 先计算总行数
            f.seek(0)  # 重置文件指针

            # 循环遍历每一行，获取所有的菜名、所有的菜系、菜名与菜系的关系、以及获取菜肴信息表
            for line in tqdm(f, total=total_lines, desc="读取数据"):
                cook_dict = {}
                data_json = json.loads(line)
                # 存放菜名（每一行肯定有菜名）
                cook = data_json['dish_name']
                dish_names.append(cook)
                cook_dict['dish_name'] = cook
                # 每一行不一定有cuisine、features、ingredients、steps（初始化，没有赋空值）
                cook_dict['features'] = ''
                cook_dict['ingredients'] = ''
                cook_dict['steps'] = ''


                if 'cuisine' in data_json:
                    cuisines.append(data_json['cuisine'])
                    rels_cuisine.append([cook, data_json['cuisine']])

                if 'features' in data_json:
                    cook_dict['features'] = data_json['features']

                if 'ingredients' in data_json:
                    cook_dict['ingredients'] = data_json['ingredients']

                if 'steps' in data_json:
                    cook_dict['steps'] = data_json['steps']

                cook_infos.append(cook_dict)

        return cook_infos, dish_names, set(cuisines), rels_cuisine

    """
        创建菜名节点，以及其他的属性
    """
    def create_dish_name_nodes(self, cook_infos):
        # 使用tqdm显示创建进度
        for cook_dict in tqdm(cook_infos, desc="创建菜名节点"):
            node = Node('Dish_Name',
                        name=cook_dict['dish_name'],
                        features=cook_dict['features'],
                        ingredients=cook_dict['ingredients'],
                        steps=cook_dict['steps'])
            self.g.create(node)

    """
        创建菜系节点
    """
    def create_cuisine_node(self, label, cuisines):
        # 使用tqdm显示创建进度
        for cuisine in tqdm(cuisines, desc="创建菜系节点"):
            node = Node(label, name=cuisine)
            self.g.create(node)

    """
        创建关系
    """
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        # 使用tqdm显示创建进度
        for edge in tqdm(edges, desc=f"创建{rel_type}关系"):
            p = edge[0]
            q = edge[1]
            query = f"MATCH (p:{start_node}), (q:{end_node}) WHERE p.name='{p}' AND q.name='{q}' CREATE (p)-[rel:{rel_type} {{name:'{rel_name}'}}]->(q)"
            try:
                self.g.run(query)
            except Exception as e:
                print(f"创建关系失败: {e}")
    """
        创建（节点、关系）总方法
    """
    def create_graph(self):
        cook_infos, dish_names, cuisines, rels_cuisine = self.read_nodes()
        self.create_dish_name_nodes(cook_infos)
        self.create_cuisine_node('Cuisines', cuisines)
        self.create_relationship('Dish_Name', 'Cuisines', rels_cuisine, 'belongs_to', '所属菜系')

    """
        导出数据
    """
    def export_data(self):
        cook_infos, dish_names, cuisines, _ = self.read_nodes()
        with open('./data/dish_name.txt', 'w+', encoding='utf-8') as f_dish_name, \
                open('./data/cuisines.txt', 'w+', encoding='utf-8') as f_cuisines:
            f_dish_name.write('\n'.join(list(dish_names)))
            f_cuisines.write('\n'.join(list(cuisines)))


if __name__ == '__main__':
    handler = CookGraph()
    handler.create_graph()
    handler.export_data()