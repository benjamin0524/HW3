import torch
from torch import nn
import numpy as np
import pickle
import utils
import os
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import math


X_LEN = 512  
EPOCHS = 50  
SAVE_INTERVAL = 10  


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path', type=str, help='the dictionary path.', default='./dictionary.pkl')
    parser.add_argument('--device', type=str, help='gpu device.', default='cuda')
    parser.add_argument('--ckp_folder', type=str, help='checkpoint folder.', default='./checkpoints_transformer')
    parser.add_argument('--mode', type=str, help='train or test_unconditional', default='train', choices=['train', 'test_unconditional'])
    parser.add_argument('--model_path', type=str, help='model path for testing', default='')
    parser.add_argument('--output_dir', type=str, help='output directory for generation', default='./results')
    parser.add_argument('--n_target_bar', type=int, help='number of bars to generate', default=32)
    parser.add_argument('--temperature', type=float, help='temperature for sampling', default=1.2)
    parser.add_argument('--topk', type=int, help='top-k for sampling', default=5)
    parser.add_argument('--n_samples', type=int, help='number of samples to generate', default=20)
    parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPUs with DataParallel')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0,1,2')
    args = parser.parse_args()
    return args

class MusicDataset(Dataset):
    def __init__(self, midi_l = [], dict_path=None):
        self.midi_l = midi_l
        self.x_len = X_LEN
        self.dictionary_path = dict_path if dict_path else opt.dict_path
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.midi_l)
    
    def __len__(self):
        return len(self.parser)  
    
    def __getitem__(self, index):
        return self.parser[index]
    
    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
        
    def prepare_data(self, midi_paths):
        all_events = []
        print("正在提取事件...")
        for path in tqdm(midi_paths):  
            try:
                events = self.extract_events(path)
                all_events.append(events)
            except Exception as e:
                print(f"处理 {path} 时出错: {e}")
                continue
        
        all_words = []
        print("正在转换为 tokens...")
        for events in tqdm(all_events):
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    if event.name == 'Note Velocity':
                        if 'Note Velocity_21' in self.event2word:
                            words.append(self.event2word['Note Velocity_21'])
                        else:
                            velocity_keys = [k for k in self.event2word.keys() if 'Note Velocity' in k]
                            if velocity_keys:
                                words.append(self.event2word[velocity_keys[0]])
                    else:
                        print(f'something is wrong! {e}')
            all_words.append(words)
        
        segments = []
        for words in all_words:
            step = self.x_len // 2
            for i in range(0, len(words) - self.x_len, step):
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                segments.append([x, y])
        
        print(f"数据段数量: {len(segments)}")
        return segments

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_head=8, n_layer=12, d_ff=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=X_LEN)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 创建 causal mask (防止看到未来的信息)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        output = self.fc_out(x)
        
        return output

def temperature_sampling(logits, temperature, topk):
    logits = logits / temperature
    if topk > 0:
        topk_logits, topk_indices = torch.topk(logits, min(topk, len(logits)), dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_logits)
        logits = mask
    
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()
    
    word = torch.multinomial(probs, 1).item()
    return word


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def test_unconditional(n_target_bar=32, temperature=1.2, topk=5, output_path='', model_path='', n_samples=1, dict_path=None):
    dict_path = dict_path if dict_path else './dictionary.pkl'
    event2word, word2event = pickle.load(open(dict_path, 'rb'))  # 加载字典，event2word和word2event的映射

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置设备

    with torch.no_grad():
        # 加载模型checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        vocab_size = checkpoint.get('vocab_size', len(event2word))
        model = TransformerModel(vocab_size=vocab_size).to(device)  # 初始化模型
        
        # 处理模型的state_dict（去除module前缀，支持DataParallel）
        state_dict = checkpoint['model']
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)  # 加载模型权重
        model.eval()  # 设置模型为评估模式

        for sample_idx in range(n_samples):
            words = [event2word['Bar_None']]  # 每次生成的音符序列初始化为一个空小节（Bar_None）

            # 随机选一个节奏（Tempo Class 和 Tempo Value）来初始化
            tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
            tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
            if tempo_classes and tempo_values:
                words.append(event2word['Position_1/16'])  # 添加1/16位置标记
                words.append(np.random.choice(tempo_classes))  # 随机选择一个Tempo Class
                words.append(np.random.choice(tempo_values))  # 随机选择一个Tempo Value

            current_generated_bar = 0
            max_tokens = n_target_bar * 500  # 限制最大token数
            token_count = 0

            while current_generated_bar < n_target_bar and token_count < max_tokens:
                token_count += 1
                if token_count % 100 == 0:
                    print(f"\r样本 {sample_idx+1}/{n_samples} - 已生成 {current_generated_bar}/{n_target_bar} bars, tokens: {token_count}", end="")

                # 提取最后的X_LEN个tokens作为输入
                temp_x = torch.tensor([words[-X_LEN:]], dtype=torch.long).to(device)
                output_logits = model(temp_x)  # 获取模型输出的logits

                # 取最后一个位置的logits来决定下一个生成的token
                _logit = output_logits[0, -1, :].cpu().detach()
                word = temperature_sampling(_logit, temperature, topk)  # 使用温度采样法来决定下一个token
                words.append(word)  # 将生成的word添加到当前序列中

                if word == event2word['Bar_None']:  # 如果生成了Bar_None，表示一个小节已经完成
                    current_generated_bar += 1

            # 为每个样本生成不同的文件名
            base_name = os.path.splitext(output_path)[0]
            ext = os.path.splitext(output_path)[1]
            if n_samples > 1:
                output_file = f"{base_name}_{sample_idx+1}{ext}"
            else:
                output_file = output_path

            # 使用utils模块中的write_midi方法将token序列转换为MIDI文件并保存
            utils.write_midi(words=words, word2event=word2event, output_path=output_file)
            print(f"\n已保存到: {output_file}")

def main():
    global opt
    opt = parse_opt()
    
    if opt.mode == 'train':
        train()
    elif opt.mode == 'test_unconditional':
        os.makedirs(opt.output_dir, exist_ok=True)
        output_path = os.path.join(opt.output_dir, 'generated.mid')
        test_unconditional(
            n_target_bar=opt.n_target_bar,
            temperature=opt.temperature,
            topk=opt.topk,
            output_path=output_path,
            model_path=opt.model_path,
            n_samples=opt.n_samples,
            dict_path=opt.dict_path
        )
    else:
        print(f"未知模式: {opt.mode}")

if __name__ == '__main__':
    main()
