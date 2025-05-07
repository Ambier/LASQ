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
    train_dir = os.path.join(output_dir, 'train_txt_eval')
    val_dir = os.path.join(output_dir, 'val_txtt_eval')
    test_dir = os.path.join(output_dir, 'test_txtt_eval')
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

            prompt_data = f"""
你是一个负责评估考生答案的智能助理。以下是你的工作要求：
1、根据考题和参考答案，分析考生答案的整体表现，给出明确的评价（如“好”“一般”“不好”）。
2、在评价中，指出考生答案的优点（正确点）和缺点（错误点），帮助考官了解答案的关键问题。

具体操作与输出格式要求：
1、提供考题、评分标准、参考答案和考生答案。
2、根据评分标准提取答案中的正确点和错误点，并结合这些点给出整体评价。
3、输出格式：一段简短的描述，尽量不超过200个汉字。不需要分点描述
4、输出样例如下：
{"考生答案整体评价为“较好”。优点：详细列出了短信渠道违规的具体情形，如误导性红包信息、不当使用品牌词等，符合评分标准中提到的违规事实。同时，考生也提到了会依据违规事实进行处罚，并建议考生自查，体现了平台处理问题的严谨性和公正性。缺点：考生答案中未提及处罚的谨慎性和多次复核的过程，且对不能提供个人证据的解释不够充分，可能影响考生对处罚合理性的理解。"}

试题如下：
{title}

评分标准如下：
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
key_result_path = "./api_processed_dataset"
process_csv_and_save_results(combine_csv_path, key_result_path)
