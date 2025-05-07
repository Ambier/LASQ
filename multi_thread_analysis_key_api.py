import csv
import os
import requests
import json

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


def process_csv_and_save_results(input_csv, output_dir):
    """
    从CSV文件读取数据，调用API处理，并将结果保存到指定文件夹中。
    """
    # 创建文件夹
    train_dir = os.path.join(output_dir, 'train_key_analysis')
    val_dir = os.path.join(output_dir, 'val_key_analysis')
    test_dir = os.path.join(output_dir, 'test_key_analysis')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 读取CSV文件
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            unique_id = row['ID']
            title = row['Title']
            dataset = row['Set']
            answer = row['Answer']
            analysis = row['Analysis']

            # 确定输出文件路径
            if dataset == 'train':
                output_path = os.path.join(train_dir, f"{unique_id}.txt")
            elif dataset == 'val':
                output_path = os.path.join(val_dir, f"{unique_id}.txt")
            elif dataset == 'test':
                output_path = os.path.join(test_dir, f"{unique_id}.txt")
            else:
                continue  # 无效数据集标识

            # 检查文件是否已存在
            if os.path.exists(output_path):
                # print(f"文件已存在，跳过：{output_path}")
                # continue
                with open(output_path, 'r', encoding='utf-8') as outfile:
                        content = outfile.read().strip()
                # 如果内容是报错信息，则重新处理
                if content == 'Wrong request, No output':
                    print(f"检测到错误内容，重新处理文件：{output_path}")
                else:
                    print(f"文件已存在，跳过：{output_path}")
                    continue
            
            # 构造prompt信息
            prompt_data = f"""
你是一个考试智能助理，现在需要帮助考官从参考答案中提取关键知识点。以下是具体要求：
1、参考答案可能包含冗余信息或过于详细的描述，请只关注其中的关键知识点。
2、根据考题的参考答案和评分标准，从参考答案中提取核心知识点。
3、每个知识点尽量简洁，最好控制在25个汉字以内，避免过长。
4、提取的知识点应直接反映考题的核心内容，避免冗余描述。
5、每个知识点应尽量规范化，确保其简洁明了，最大限度避免无关或重复信息。
6、最多提取2-3个知识点，如果不足，请忽略。

具体操作：
1、提供考题和参考答案，请从参考答案中抽取关键知识点。
2、得分点需要用简洁的中文列出，按出现顺序排列。
3、输出格式应遵循标准要求，避免任何不相关内容。
4、如果答案足够简短可以直接作为知识点，则直接输出原答案即可。

输出格式要求：
1、每个知识点用分号隔开，并按顺序排列。

试题如下：
{title}

参考答案如下：
{analysis}
            """

            try:
                # 调用API处理
                result = attain_result(prompt_data, gpt_type="qwen2.5-72b-instruct", temperature=0.75)
                # 将结果写入文件
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(result)
                print(f"成功处理并保存：{output_path}")
            except Exception as e:
                print(f"处理失败（跳过）：ID={unique_id}, Error={e}")

# 示例用法
combine_csv_path = "./train_valid_test.csv"
key_result_path = "./api_processed_dataset"
process_csv_and_save_results(combine_csv_path, key_result_path)
