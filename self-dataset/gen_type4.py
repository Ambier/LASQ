## 用于生成：基础教育，语数英相关内容

import csv
import os
import requests
import json

# 构造知识点的prompt内容
# 人文相关：公益、文艺类型知识点层次结构

knowledge_points_humanities = {
    '人文相关': {
        '公益': {
            '公益概念与理念': [
                '公益的定义与类型',
                '公益价值观（利他主义、社会责任）',
                '可持续发展理念'
            ],
            '公益项目管理': [
                '项目策划与设计',
                '项目执行与监督',
                '绩效评估与反馈'
            ],
            '志愿者服务': [
                '志愿者的权利与义务',
                '志愿服务技能培训',
                '志愿者激励与管理'
            ],
            '社会问题与需求': [
                '贫困与救助',
                '教育公平',
                '环境保护与生态文明'
            ],
            '公益传播与筹款': [
                '公益广告与宣传',
                '社交媒体的应用',
                '筹款方法与策略'
            ],
            '法律法规与伦理': [
                '慈善法与公益组织管理条例',
                '公益活动的合法性与合规性',
                '公益伦理道德'
            ]
        },
        '文艺类型': {
            '文学': [
                '文学体裁（小说、诗歌、散文、戏剧）',
                '文学流派与思潮（现实主义、浪漫主义）',
                '文学创作技巧（描写、叙述、抒情）'
            ],
            '艺术': [
                '视觉艺术（绘画、雕塑、摄影）',
                '表演艺术（音乐、舞蹈、戏剧）',
                '现代艺术形式（装置艺术、数字艺术）'
            ],
            '文化与历史': [
                '中国传统文化（儒释道思想、传统节日）',
                '世界文化遗产与名胜古迹',
                '文化多样性与全球化'
            ],
            '艺术鉴赏': [
                '艺术作品的欣赏方法',
                '艺术评论与批评',
                '审美理论与美学观点'
            ],
            '文化产业': [
                '文化创意产业概念',
                '文化产品的开发与运营',
                '版权保护与知识产权'
            ],
            '传媒与传播': [
                '传统媒体与新媒体',
                '文化传播的渠道与方式',
                '媒体素养与信息辨识'
            ],
            '创意写作': [
                '故事构思与情节设计',
                '人物塑造与对话编写',
                '文学语言的运用'
            ]
        }
    }
}

import random

# 人文相关知识点层次结构（如上所述）

def generate_humanities_prompts(knowledge_structure, total_questions):
    prompts = []
    # 计算总的知识点数量
    total_knowledge_points = 0
    for theme in knowledge_structure['人文相关']:
        for module in knowledge_structure['人文相关'][theme]:
            total_knowledge_points += len(knowledge_structure['人文相关'][theme][module])

    # 计算每个知识点应分配的题目数量
    questions_per_knowledge_point = total_questions // total_knowledge_points

    # 开始生成prompt
    for theme in knowledge_structure['人文相关']:
        for module in knowledge_structure['人文相关'][theme]:
            for knowledge_point in knowledge_structure['人文相关'][theme][module]:
                for _ in range(questions_per_knowledge_point):
                    prompt = create_humanities_prompt('人文相关', theme, module, knowledge_point)
                    prompts.append(prompt)

    # 如果还有剩余的题目，随机分配
    remaining_questions = total_questions - len(prompts)
    if remaining_questions > 0:
        additional_prompts = []
        for _ in range(remaining_questions):
            # 随机选择一个知识点
            theme = random.choice(list(knowledge_structure['人文相关'].keys()))
            module = random.choice(list(knowledge_structure['人文相关'][theme].keys()))
            knowledge_point = random.choice(knowledge_structure['人文相关'][theme][module])
            prompt = create_humanities_prompt('人文相关', theme, module, knowledge_point)
            additional_prompts.append(prompt)
        prompts.extend(additional_prompts)

    return prompts


# 给予模板构造prompt
def create_humanities_prompt(category, theme, module, knowledge_point):
    prompt_template = (
        f"请根据以下知识点出一道题目。\n"
        f"类别：{category}\n"
        f"主题：{theme}\n"
        f"模块：{module}\n"
        f"知识点：{knowledge_point}\n"
        f"要求：题目具有专业性，内容准确，具有针对性，并提供标准答案。"
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
    total_questions_humanities = 1800  # 包括训练集、验证集和测试集

    # 保存为CSV文件
    csv_file_path = "/type4_human_knowledge.csv"

    if not os.path.exists(csv_file_path):
        # 生成所有的prompt
        all_humanities_prompts = generate_humanities_prompts(knowledge_points_humanities, total_questions_humanities)

        # 打乱顺序
        random.shuffle(all_humanities_prompts)

        # 将生成的prompt分为训练集、验证集和测试集
        train_prompts = all_humanities_prompts[:1500]
        valid_prompts = all_humanities_prompts[1500:1650]
        test_prompts = all_humanities_prompts[1650:1800]

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
    store_path = "/type4"
    process_csv_and_generate_results(csv_file_path, store_path)

