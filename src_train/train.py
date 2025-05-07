import os

import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import logging
logger = logging.getLogger()

from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from model import *
import random

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LogCoshLoss(nn.Module):
    def forward(self, inputs, targets):
        logcosh = torch.log(torch.cosh(inputs - targets))
        return torch.mean(logcosh)

if torch.cuda.is_available():
    print('Hello my friends')

transformers.logging.set_verbosity_error()

class TextSimilarityDataset(Dataset):  # eval_senti qwen_eval answer_key ref_key
    def __init__(self, file_path, bert_tokenizer, data_type='val'):
        self.tokenizer = bert_tokenizer
        # 读取 CSV 文件并丢弃任何包含 NaN 的行，同时重置索引
        self.data = pd.read_csv(file_path).dropna(subset=[
            'title', 'fake_query', 'answer', 'analysis', 'eval', 'answer_key', 'ref_key', 'similarity'
        ]).reset_index(drop=True)
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            title = self.data.loc[index, 'title']
            fake_query = self.data.loc[index, 'fake_query']
            answer = self.data.loc[index, 'answer']
            analysis = self.data.loc[index, 'analysis']
            qwen_eval = self.data.loc[index, 'eval']
            answer_key = self.data.loc[index, 'answer_key']
            ref_key = self.data.loc[index, 'ref_key']
            similarity = self.data.loc[index, 'similarity']

            key_pair = self.tokenizer.encode_plus(
                answer_key,
                ref_key,
                add_special_tokens=True,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt'
            )

            pair_sentence = self.tokenizer.encode_plus(
                answer,
                analysis,
                add_special_tokens=True,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt'
            )

            question = self.tokenizer.encode_plus(
                title,
                fake_query,
                add_special_tokens=True,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt'
            )

            qwen_eval = self.tokenizer(
                qwen_eval,
                add_special_tokens=True,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt'
            )

            return {
                'qwen_eval': qwen_eval,
                'question': question,
                'key_pair': key_pair,
                'pair_sentence': pair_sentence,
                'similarity': torch.tensor(similarity, dtype=torch.float)
            }

        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise


    
def train(save_path, model, train_loader, valid_loader, device, num_epochs=5, learning_rate=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_data = 0

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for batch in train_iterator:
            
            qwen_eval = batch['qwen_eval'].to(device)
            question = batch['question'].to(device)
            key_pair = batch['key_pair'].to(device)
            pair_sentence = batch['pair_sentence'].to(device)
            # eval_senti = batch['eval_senti'].to(device)
            similarity = batch['similarity'].to(device)
            optimizer.zero_grad() # 清理之前梯度

            overall_similarity = model(qwen_eval, question, key_pair, pair_sentence)
            predicted_similarity = overall_similarity
            loss = criterion(predicted_similarity, similarity)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # 更新tqdm进度条
            num_data += len(batch)
            train_iterator.set_postfix({'train_loss': train_loss / num_data})

        train_loss /= len(train_loader)
        weights_path = os.path.join(save_path, "bertchinese_{}_weights.pt".format(str(epoch+1)))
        torch.save(model.state_dict(), weights_path)

        
        model.eval()
        valid_loss = 0.0
        total_valid_loss = float('inf')

        for batch in valid_loader:
            qwen_eval = batch['qwen_eval'].to(device)
            question = batch['question'].to(device)
            key_pair = batch['key_pair'].to(device)
            pair_sentence = batch['pair_sentence'].to(device)
            similarity = batch['similarity'].to(device)

            with torch.no_grad():
                predicted_similarity = model(qwen_eval, question, key_pair, pair_sentence)
                loss = criterion(predicted_similarity, similarity)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        logger.info("Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, ".format(
                    epoch+1,
                    train_loss,
                    valid_loss
                ))
        
if_local = True
if if_local: 

    # 业务数据
    train_path = '/train.csv'
    valid_path = '/valid.csv'
    save_path = '/overall_cross_attention_promax' 

    if not os.path.exists(save_path): os.mkdir(save_path)
    
    # 确定模型名称
    bert_model_name = './bert-base-chinese'

log_path = os.path.join(save_path ,'train_log.txt')
handlers = [logging.FileHandler(log_path)]
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                    datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
logging.getLogger('PIL').setLevel(logging.WARNING)

#------------------------------修改权重保存文件夹-------------------------------------
#BERT模型
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = XiaokaoLLM_promax(bert_model)  
batchsize = 20 

train_dataset = TextSimilarityDataset(train_path, bert_tokenizer, data_type='train')
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

valid_dataset = TextSimilarityDataset(valid_path, bert_tokenizer, data_type='val')
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

model.to(device)
train(save_path, model, train_loader, valid_loader, device, num_epochs=10, learning_rate=3e-5) 

print(save_path)


