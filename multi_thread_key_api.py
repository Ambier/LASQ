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
    train_dir = os.path.join(output_dir, 'train_txt')
    val_dir = os.path.join(output_dir, 'val_txt')
    test_dir = os.path.join(output_dir, 'test_txt')
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

            # 构造prompt模板
            prompt_data = f"""
你是一个智能助理，现在需要帮助考官对考生的答案进行评分。以下是具体要求：
1、考生答案可能包含大量无关信息或错误内容，但你需要只关注其中的得分点。
2、根据考题的参考答案和评分标准，从考生答案中提取关键知识点。
3、如果考生答案中没有任何得分点，请直接标注“无得分点”。
4、请忽略冗余信息，不要对答案进行主观评价，仅提取关键知识点，可适当凝练信息，但不可改变原意。
5、仅需抽取2-3个知识点，数量不足可忽略。
6、每个知识点，尽量规范在25个汉字以内，避免内容过长

具体操作：
1、考题、评分标准和考生答案将分别提供。请从考生答案中抽取得分点。
2、得分点需要用简洁的中文列出，按出现顺序排列。
3、如果有多个得分点，请用列表的形式输出。

输出格式要求：
1、仅输出最终提取的知识点，避免其他不相关或冗余的表达
2、输出样例(每个关键词用分号隔开，避免任何不相关内容):
{"知识点一;知识点二;知识点三"}
3、若没有知识点，请直接输出："无得分点"

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
key_result_path = "./api_processed_dataset"
process_csv_and_save_results(combine_csv_path, key_result_path)
