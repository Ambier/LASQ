## 用于生成：基础教育，语数英相关内容

import csv
import os
import requests
import json

# 构造知识点的prompt内容
knowledge_points = {
    '基础教育': {
        '低年级': {
            '语文': {
                '拼音与发音': [
                    '声母、韵母、整体认读音节',
                    '四声的认读和标注',
                    '音节的拼读规则'
                ],
                '汉字认读与书写': [
                    '常用汉字的笔画和笔顺',
                    '部首认识与分类',
                    '简单汉字的书写练习',
                    '偏旁部首的含义及其在字义上的作用'
                ],
                '词语': [
                    '常用词语的积累与运用',
                    '近义词、反义词的辨析',
                    '量词的正确使用',
                    '词语搭配和组词练习'
                ],
                '句子': [
                    '简单句的构成（主语、谓语、宾语）',
                    '标点符号的使用（句号、逗号、问号、感叹号）',
                    '陈述句、疑问句、感叹句的区别',
                    '造句练习'
                ],
                '阅读理解': [
                    '顺序、因果关系的理解'
                ],
                '写话与作文': [
                    '描写人物、动物、景物的基本方法',
                    '写作中的时间、地点、人物、事件要素'
                ],
                '听说能力': [
                    '认真倾听他人讲话',
                    '复述简单的故事',
                    '口头表达清晰、连贯',
                    '礼貌用语的使用'
                ],
                '古诗词背诵': [
                    '常见儿童古诗的背诵与理解',
                    '诗句的朗读技巧',
                    '简单的诗意解释'
                ]
            },
            '数学': {
                '数与计算': [
                    '认识数字0-100',
                    '顺数与倒数',
                    '比较大小（大于、小于、等于）'
                ],
                '量的认识': [
                    '时间的认识（整点、半点）',
                    '长度单位（厘米、米）的初步认识',
                    '重量单位（克、千克）的初步认识',
                    '简单的测量工具使用'
                ],
                '图形与几何': [
                    '平面图形的认识（长方形、正方形、三角形、圆形）',
                    '立体图形的认识（正方体、长方体、圆柱、球）',
                    '图形的基本特征和简单分类',
                    '图形的拼组和分割'
                ],
                '应用题': [
                    '简单的文字题理解',
                    '根据问题选择正确的运算方法',
                    '解答一步计算的应用题'
                ],
                '规律与排序': [
                    '简单的数字、图形规律',
                    '按规律填数或画图',
                    '排序练习（大小、长短、高矮）'
                ],
                '数据处理': [
                    '简单的统计表和条形图',
                    '收集和整理数据',
                    '数据的比较和描述'
                ],
                '钱币的认识': [
                    '认识人民币的面值'
                ],
                '空间与方向': [
                    '上下、前后、左右的辨别',
                    '简单路线的描述',
                    '地图的初步认识'
                ]
            },
            '英语': {
                '字母与发音': [
                    '基本的字母发音规则',
                    '简单的字母组合发音'
                ],
                '基础词汇': [
                    '日常用语（问候、告别、感谢）',
                    '颜色、数字、天气、时间等词汇',
                    '家庭成员、动物、食物、物品名称'
                ],
                '基本句型': [
                    '简单的陈述句（I am...，You are...）',
                    '一般疑问句（Are you...？，Is this...?）',
                    '肯定与否定回答（Yes, I am. / No, I am not.）'
                ],
                '听力与口语': [
                    '听懂简单的指令和问题',
                    '模仿正确的语音语调',
                    '进行简单的自我介绍',
                    '简单的对话练习'
                ],
                '阅读与理解': [
                    '识别单词和简单句子',
                    '理解常见标识和标志的含义',
                    '阅读短小的故事或段落'
                ],
                '书写与拼写': [
                    '正确书写字母和单词',
                    '抄写简单的句子',
                    '拼写常用单词'
                ],
                '语法基础': [
                    '名词的单复数形式',
                    '常用动词的简单现在时',
                    '人称代词（I, you, he, she, it, we, they）'
                ],
                '文化知识': [
                    '英语国家的简单风俗习惯',
                    '常见的英语儿歌和童谣'
                ]
            }
        }
    }
}



import random
# 生成相关prompt内容
def generate_prompts(knowledge_structure, total_questions):
    prompts = []
    # 计算总的知识点数量
    total_knowledge_points = 0
    for subject in knowledge_structure['基础教育']['低年级']:
        for topic in knowledge_structure['基础教育']['低年级'][subject]:
            total_knowledge_points += len(knowledge_structure['基础教育']['低年级'][subject][topic])
    
    # 计算每个知识点应分配的题目数量
    questions_per_knowledge_point = total_questions // total_knowledge_points
    
    # 开始生成prompt
    for subject in knowledge_structure['基础教育']['低年级']:
        for topic in knowledge_structure['基础教育']['低年级'][subject]:
            for knowledge_point in knowledge_structure['基础教育']['低年级'][subject][topic]:
                for _ in range(questions_per_knowledge_point):
                    prompt = create_prompt('基础教育', '低年级', subject, topic, knowledge_point)
                    prompts.append(prompt)
    
    # 如果还有剩余的题目，随机分配
    remaining_questions = total_questions - len(prompts)
    if remaining_questions > 0:
        additional_prompts = []
        for _ in range(remaining_questions):
            # 随机选择一个知识点
            subject = random.choice(list(knowledge_structure['基础教育']['低年级'].keys()))
            topic = random.choice(list(knowledge_structure['基础教育']['低年级'][subject].keys()))
            knowledge_point = random.choice(knowledge_structure['基础教育']['低年级'][subject][topic])
            prompt = create_prompt('基础教育', '低年级', subject, topic, knowledge_point)
            additional_prompts.append(prompt)
        prompts.extend(additional_prompts)
    
    return prompts

# 给予模板构造prompt
def create_prompt(category, level, subject, topic, knowledge_point):
    prompt_template = (
        f"请根据以下知识点出一道题目。\n"
        f"类别：{category}\n"
        f"学段：{level}\n"
        f"科目：{subject}\n"
        f"主题：{topic}\n"
        f"知识点：{knowledge_point}\n"
        f"要求：题目符合低年级学生的理解水平，内容准确，具有针对性，并提供正确答案。"
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
    total_questions = 4200  # 包括训练集、验证集和测试集

    # 保存为CSV文件
    csv_file_path = "type_school_knowledge.csv"

    if not os.path.exists(csv_file_path):
        # 生成所有的prompt
        all_prompts = generate_prompts(knowledge_points, total_questions)

        # 将生成的prompt分为训练集、验证集和测试集
        train_prompts = all_prompts[:3500]
        valid_prompts = all_prompts[3500:3850]
        test_prompts = all_prompts[3850:4200]

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
    store_path = "/type1"
    process_csv_and_generate_results(csv_file_path, store_path)

