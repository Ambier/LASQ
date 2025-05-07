## 用于生成：基础教育，语数英相关内容

import csv
import os
import requests
import json

# 构造知识点的prompt内容
# 计算机：计算机一级/二级知识点层次结构

knowledge_points_computer = {
    '计算机': {
        '计算机一级': {
            '计算机基础知识': [
                '计算机的组成与工作原理',
                '数制与编码（二进制、ASCII码）',
                '操作系统基础（Windows常用操作）'
            ],
            '办公软件应用': {
                'Word文字处理': [
                    '文档编辑与排版',
                    '图文混排',
                    '页面设置与打印'
                ],
                'Excel电子表格': [
                    '数据输入与格式设置',
                    '图表制作'
                ],
                'PowerPoint演示文稿': [
                    '幻灯片的创建与设计',
                    '动画效果与切换',
                    '放映设置与演示技巧'
                ]
            },
            '网络基础': [
                '因特网基础知识',
                '浏览器的使用与网页浏览',
                '电子邮件的收发与管理'
            ],
            '信息安全与道德': [
                '计算机病毒防护',
                '密码安全与个人信息保护',
                '网络道德规范'
            ]
        },
        '计算机二级': {
            '高级办公软件应用': {
                '高级Word': [
                    '样式与模板应用',
                    '长文档编辑（目录、脚注、引用）',
                    '邮件合并'
                ],
                '高级Excel': [
                    '函数嵌套与数组公式',
                    '数据透视表与数据分析',
                    '宏与简单VBA编程'
                ],
                '高级PowerPoint': [
                    '母版与主题设计',
                    '多媒体元素的嵌入',
                    '演示文稿的包装与发布'
                ]
            },
            '程序设计': {
                'C语言程序设计': [
                    '数据类型与运算符',
                    '控制结构（条件语句、循环语句）',
                    '数组、指针与函数',
                    '文件操作'
                ],
                'Java语言程序设计': [
                    '面向对象编程概念（类与对象、继承、多态）',
                    '基本语法与数据类型',
                    '异常处理',
                    '简单GUI编程'
                ],
                'Python语言程序设计': [
                    '基本数据类型与结构（列表、字典）',
                    '控制流与函数',
                    '模块与包的使用',
                    '文件读写'
                ]
            },
            '数据库应用': [
                '数据库基础理论（关系模型、主键、外键）',
                'SQL语法与查询（SELECT、INSERT、UPDATE、DELETE）',
                '数据库的创建与管理',
                '简单数据库应用程序开发'
            ],
            '计算机网络': [
                '网络协议基础（TCP/IP、HTTP）',
                '局域网基础知识',
                '网络设备与拓扑结构'
            ],
            '软件工程': [
                '软件开发生命周期',
                '需求分析与设计',
                '测试与维护'
            ]
        }
    }
}

import random

# 计算机知识点层次结构（如上所述）

def generate_computer_prompts(knowledge_structure, total_questions):
    prompts = []
    # 计算总的知识点数量
    total_knowledge_points = 0
    for level in knowledge_structure['计算机']:
        for module in knowledge_structure['计算机'][level]:
            if isinstance(knowledge_structure['计算机'][level][module], dict):
                for sub_module in knowledge_structure['计算机'][level][module]:
                    total_knowledge_points += len(knowledge_structure['计算机'][level][module][sub_module])
            else:
                total_knowledge_points += len(knowledge_structure['计算机'][level][module])
    
    # 计算每个知识点应分配的题目数量
    questions_per_knowledge_point = total_questions // total_knowledge_points
    
    # 开始生成prompt
    for level in knowledge_structure['计算机']:
        for module in knowledge_structure['计算机'][level]:
            if isinstance(knowledge_structure['计算机'][level][module], dict):
                # 有子模块的情况
                for sub_module in knowledge_structure['计算机'][level][module]:
                    for knowledge_point in knowledge_structure['计算机'][level][module][sub_module]:
                        for _ in range(questions_per_knowledge_point):
                            prompt = create_computer_prompt('计算机', level, module, sub_module, knowledge_point)
                            prompts.append(prompt)
            else:
                # 没有子模块的情况
                for knowledge_point in knowledge_structure['计算机'][level][module]:
                    for _ in range(questions_per_knowledge_point):
                        prompt = create_computer_prompt('计算机', level, module, None, knowledge_point)
                        prompts.append(prompt)
    
    # 如果还有剩余的题目，随机分配
    remaining_questions = total_questions - len(prompts)
    if remaining_questions > 0:
        additional_prompts = []
        for _ in range(remaining_questions):
            # 随机选择一个知识点
            level = random.choice(list(knowledge_structure['计算机'].keys()))
            module = random.choice(list(knowledge_structure['计算机'][level].keys()))
            if isinstance(knowledge_structure['计算机'][level][module], dict):
                sub_module = random.choice(list(knowledge_structure['计算机'][level][module].keys()))
                knowledge_point = random.choice(knowledge_structure['计算机'][level][module][sub_module])
            else:
                sub_module = None
                knowledge_point = random.choice(knowledge_structure['计算机'][level][module])
            prompt = create_computer_prompt('计算机', level, module, sub_module, knowledge_point)
            additional_prompts.append(prompt)
        prompts.extend(additional_prompts)
    
    return prompts


# 给予模板构造prompt
def create_computer_prompt(category, level, module, sub_module, knowledge_point):
    if sub_module:
        prompt_template = (
            f"请根据以下知识点出一道题目。\n"
            f"类别：{category}\n"
            f"等级：{level}\n"
            f"模块：{module}\n"
            f"子模块：{sub_module}\n"
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
    else:
        prompt_template = (
            f"请根据以下知识点出一道题目。\n"
            f"类别：{category}\n"
            f"等级：{level}\n"
            f"模块：{module}\n"
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
    total_questions_computer = 3000  # 包括训练集、验证集和测试集

    # 保存为CSV文件
    csv_file_path = "/type3_compute_knowledge.csv"

    if not os.path.exists(csv_file_path):
        # 生成所有的prompt
        all_computer_prompts = generate_computer_prompts(knowledge_points_computer, total_questions_computer)

        # 打乱顺序
        random.shuffle(all_computer_prompts)

        # 将生成的prompt分为训练集、验证集和测试集
        train_prompts = all_computer_prompts[:2500]
        valid_prompts = all_computer_prompts[2500:2750]
        test_prompts = all_computer_prompts[2750:3000]

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
    store_path = "/type3"
    process_csv_and_generate_results(csv_file_path, store_path)

