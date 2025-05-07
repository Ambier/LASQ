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
    train_dir = os.path.join(output_dir, 'train_query')
    val_dir = os.path.join(output_dir, 'val_query')
    test_dir = os.path.join(output_dir, 'test_query')
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

            # 构造prompt
            prompt_data = f"""
你是一个考试智能助理，需要通过考生的答案生成伪考题，并将其与实际考题进行对比，以判断考生答案的相关性和准确性。以下是具体要求：

根据考生答案生成伪考题：
1、从考生答案中提取核心问题或描述。
2、忽略冗余信息，避免过度关注错误点。
3、伪考题应与考生答案的主要语义一致。

输出格式：
1、直接提供生成的伪考题，避免冗余内容，不要有多余描述
2、输出举例（直接输出内容）：{"伪考题"}

试题如下：
{title}

参考答案如下：
{analysis}

考生答案如下：
{answer}
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
key_result_path = "/api_processed_dataset"
process_csv_and_save_results(combine_csv_path, key_result_path)
