## 用于生成：基础教育，语数英相关内容

import csv
import os
import requests
import json

# 构造知识点的prompt内容
# 建筑工程：一/二级建筑师知识点层次结构

knowledge_points_architecture = {
    '建筑工程': {
        '一/二级建筑师': {
            '建筑设计': {
                '设计原理': [
                    '建筑美学与形式',
                    '功能与空间布局',
                    '人性化设计原则'
                ],
                '建筑构造': [
                    '基础、墙体、楼板、屋顶构造',
                    '门窗与楼梯设计',
                    '防水、防潮与保温构造'
                ],
                '建筑结构': [
                    '结构类型（砖混、框架、剪力墙、钢结构）',
                    '结构受力分析',
                    '抗震设计原则'
                ],
                '建筑物理': [
                    '声学设计（隔声、吸声）',
                    '光学设计（采光、照明）',
                    '热工设计（热传导、热舒适）'
                ],
                '建筑设备': [
                    '给排水系统设计',
                    '暖通空调系统',
                    '电气系统与智能化'
                ],
                '绿色建筑': [
                    '可持续设计理念',
                    '节能技术应用',
                    '新能源与环保材料'
                ],
                '建筑历史与理论': [
                    '中外建筑史',
                    '经典建筑案例分析',
                    '现代建筑理论与流派'
                ]
            },
            '城乡规划': {
                '城市规划原理': [
                    '城市功能分区',
                    '交通规划与组织',
                    '公共空间与绿地系统'
                ],
                '乡村规划': [
                    '乡村振兴策略',
                    '特色小镇规划',
                    '农村基础设施建设'
                ],
                '法规与规范': [
                    '城市规划相关法律法规',
                    '建设用地规划管理',
                    '历史文化名城保护条例'
                ]
            },
            '建筑法规': {
                '建筑法': [
                    '建筑市场管理',
                    '建筑工程招标投标',
                    '建筑工程承包与分包'
                ],
                '工程建设标准强制性条文': [
                    '安全生产规定',
                    '消防设计规范',
                    '无障碍设计规范'
                ],
                '职业道德': [
                    '建筑师职业责任',
                    '诚信执业',
                    '行业自律与监管'
                ]
            },
            '施工技术与管理': {
                '施工技术': [
                    '土方工程',
                    '钢筋混凝土工程',
                    '装饰装修工程'
                ],
                '施工组织设计': [
                    '施工进度计划',
                    '资源配置与管理',
                    '质量控制与验收'
                ],
                '工程项目管理': [
                    '项目成本管理',
                    '风险管理',
                    '信息管理'
                ]
            },
            '建筑经济': {
                '工程造价': [
                    '工程量清单编制',
                    '定额计价方法',
                    '造价控制与调整'
                ],
                '投资与融资': [
                    '项目可行性研究',
                    '融资模式（PPP、BOT等）',
                    '投资回报分析'
                ]
            }
        }
    }
}


import random

# 建筑工程知识点层次结构（如上所述）

def generate_architecture_prompts(knowledge_structure, total_questions):
    prompts = []
    # 计算总的知识点数量
    total_knowledge_points = 0
    for field in knowledge_structure['建筑工程']['一/二级建筑师']:
        for topic in knowledge_structure['建筑工程']['一/二级建筑师'][field]:
            total_knowledge_points += len(knowledge_structure['建筑工程']['一/二级建筑师'][field][topic])
    
    # 计算每个知识点应分配的题目数量
    questions_per_knowledge_point = total_questions // total_knowledge_points
    
    # 开始生成prompt
    for field in knowledge_structure['建筑工程']['一/二级建筑师']:
        for topic in knowledge_structure['建筑工程']['一/二级建筑师'][field]:
            for knowledge_point in knowledge_structure['建筑工程']['一/二级建筑师'][field][topic]:
                for _ in range(questions_per_knowledge_point):
                    prompt = create_architecture_prompt('建筑工程', '一/二级建筑师', field, topic, knowledge_point)
                    prompts.append(prompt)
    
    # 如果还有剩余的题目，随机分配
    remaining_questions = total_questions - len(prompts)
    if remaining_questions > 0:
        additional_prompts = []
        for _ in range(remaining_questions):
            # 随机选择一个知识点
            field = random.choice(list(knowledge_structure['建筑工程']['一/二级建筑师'].keys()))
            topic = random.choice(list(knowledge_structure['建筑工程']['一/二级建筑师'][field].keys()))
            knowledge_point = random.choice(knowledge_structure['建筑工程']['一/二级建筑师'][field][topic])
            prompt = create_architecture_prompt('建筑工程', '一/二级建筑师', field, topic, knowledge_point)
            additional_prompts.append(prompt)
        prompts.extend(additional_prompts)
    
    return prompts

# 给予模板构造prompt
def create_architecture_prompt(category, level, field, topic, knowledge_point):
    prompt_template = (
        f"请根据以下知识点出一道题目。\n"
        f"类别：{category}\n"
        f"资格等级：{level}\n"
        f"专业领域：{field}\n"
        f"主题：{topic}\n"
        f"知识点：{knowledge_point}\n"
        f"要求：题目具有专业性，符合一/二级建筑师考试的要求，内容准确，具有针对性，并提供标准答案。"
        f"""
具体要求：
1、试题必须是中文，并且是主观/客观的问答题
2、题目的长度尽量不要超过50个汉字
3、试题与答案必须要有相关性，即从答案可能可以反推回试题。
4、答案应该清晰明确，并且总分设定在5-10分之间。
5、请你严格按照json格式进行输出，避免任何不相关的输出，输出格式模板如下：
{{
    "exam": "(此处填写试题)",
    "analysis": "(此处填写参考答案)",
    "instruct": "(此处添加试题指导，即得分点是哪些)"
    "max_score": "(给出试题的总分，总分必须和得分点对应)" 
}}
        """
    )
    return prompt_template

def process_csv_and_generate_results(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 创建存储文件夹

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["id"]
            text = row["text"]
            tag = row["tag"]

            # 判断目标文件是否已存在
            output_file_path = os.path.join(output_dir, f"{tag}_{file_id}.json")
            if os.path.exists(output_file_path):
                print(f"File {output_file_path} already exists. Skipping...")
                continue

            # 调用生成函数
            result = attain_result(text)
            try:
                # 检查是否是合法的 JSON 格式
                parsed_result = json.loads(result)
            except (TypeError, json.JSONDecodeError):
                print(f"Result for ID {file_id} is not a valid JSON. Skipping...")
                continue

            # 保存 JSON 格式内容
            with open(output_file_path, mode="w", encoding="utf-8") as json_file:
                json.dump(parsed_result, json_file, indent=4, ensure_ascii=False)
            print(f"Generated and saved: {output_file_path}")

def attain_result(prompt_data, gpt_type="qwen2.5-72b-instruct", temperature=0.75):
    try:
        data = {
        "model": gpt_type,
        "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_data},
            ],
        "temperature":temperature
        }

        url = "Use Your URL"
        headers = "Use Your SK"

        # 发送POST请求
        # try:
        response = requests.post(url, headers=headers, json=data).json()
        response = response['choices'][0]['message']['content'].strip() # 这里输出是字典格式的字符串
    except:
        response = 'Wrong request, No output'
    return response


if __name__ == "__main__":

    # 步骤一：生成prompt并保存
    # 使用举例
    # 总题目数量
    total_questions_architecture = 3000  # 包括训练集、验证集和测试集

    # 保存为CSV文件
    csv_file_path = "/type2_build_knowledge.csv"

    if not os.path.exists(csv_file_path):
        # 生成所有的prompt
        all_architecture_prompts = generate_architecture_prompts(knowledge_points_architecture, total_questions_architecture)

        # 打乱顺序
        random.shuffle(all_architecture_prompts)

        # 将生成的prompt分为训练集、验证集和测试集
        train_prompts = all_architecture_prompts[:2500]
        valid_prompts = all_architecture_prompts[2500:2750]
        test_prompts = all_architecture_prompts[2750:3000]

        # 保存prompt内容，保存为csv文件
        # 构造对应的tag
        data_with_tags = [("train", text) for text in train_prompts] + \
                        [("valid", text) for text in valid_prompts] + \
                        [("test", text) for text in test_prompts]

        
        with open(csv_file_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text", "tag"])  # 写入表头
            for idx, (tag, text) in enumerate(data_with_tags, start=1):
                writer.writerow([idx, text, tag])

    # 步骤二：读取csv文件得到prompt，并且生成json文件
    # 读取csv和保存文件夹
    store_path = "/type2"
    process_csv_and_generate_results(csv_file_path, store_path)

