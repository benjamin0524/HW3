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

# ==================== 训练配置参数 ====================
X_LEN = 512  # 输入序列长度
EPOCHS = 50  # 总 epoch 数
SAVE_INTERVAL = 10  # 每 10 个 epoch 保存一次
# =====================================================

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
        
        # 創建 causal mask (防止看到未來的信息)
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
    event2word, word2event = pickle.load(open(dict_path, 'rb'))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        vocab_size = checkpoint.get('vocab_size', len(event2word))
        model = TransformerModel(vocab_size=vocab_size).to(device)
        
        state_dict = checkpoint['model']
        state_dict = remove_module_prefix(state_dict) 
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        for sample_idx in range(n_samples):
            words = [event2word['Bar_None']]  
            tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
            tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
            if tempo_classes and tempo_values:
                words.append(event2word['Position_1/16'])  
                words.append(np.random.choice(tempo_classes))  
                words.append(np.random.choice(tempo_values))  

            current_generated_bar = 0
            max_tokens = n_target_bar * 500  
            token_count = 0
            
            while current_generated_bar < n_target_bar and token_count < max_tokens:
                token_count += 1
                if token_count % 100 == 0:
                    print(f"\r样本 {sample_idx+1}/{n_samples} - 已生成 {current_generated_bar}/{n_target_bar} bars, tokens: {token_count}", end="")

                # 取最後 X_LEN 個 tokens
                temp_x = torch.tensor([words[-X_LEN:]], dtype=torch.long).to(device)
                output_logits = model(temp_x)
                
                # 取最後一個位置的 logits
                _logit = output_logits[0, -1, :].cpu().detach()
                word = temperature_sampling(_logit, temperature, topk)
                words.append(word)
                
                if word == event2word['Bar_None']:
                    current_generated_bar += 1

            # 為每個樣本生成不同的文件名
            base_name = os.path.splitext(output_path)[0]
            ext = os.path.splitext(output_path)[1]
            if n_samples > 1:
                output_file = f"{base_name}_{sample_idx+1}{ext}"
            else:
                output_file = output_path
            
            utils.write_midi(words=words, word2event=word2event, output_path=output_file)
            print(f"\n已保存到: {output_file}")

def train():
    epochs = EPOCHS

    train_list = glob.glob('./Pop1K7/midi_analyzed/**/*.mid', recursive=True)
    if not train_list:
        train_list = glob.glob('./Pop1K7/midi_analyzed/*.mid')
    print(f'train list len = {len(train_list)}')

    train_dataset = MusicDataset(train_list, dict_path=opt.dict_path)
    
    def collate_fn(batch):
        x_list = []
        y_list = []
        for item in batch:
            x_list.append(torch.tensor(item[0], dtype=torch.long))
            y_list.append(torch.tensor(item[1], dtype=torch.long))
        
        x_lengths = [x.size(0) for x in x_list]
        y_lengths = [y.size(0) for y in y_list]
        
        if len(set(x_lengths)) == 1 and len(set(y_lengths)) == 1 and x_lengths[0] == y_lengths[0]:
            return torch.stack(x_list), torch.stack(y_list)
        else:
            max_len = max(max(x_lengths), max(y_lengths))
            x_padded = []
            y_padded = []
            for x, y in zip(x_list, y_list):
                if x.size(0) < max_len:
                    x = torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.long)])
                if y.size(0) < max_len:
                    y = torch.cat([y, torch.zeros(max_len - y.size(0), dtype=torch.long)])
                x_padded.append(x)
                y_padded.append(y)
            return torch.stack(x_padded), torch.stack(y_padded)
    
    # 多 GPU 設置
    if opt.multi_gpu and torch.cuda.device_count() > 1:
        gpu_ids = [int(x) for x in opt.gpu_ids.split(',')]
        BATCH_SIZE = 32 * len(gpu_ids)  # 根據 GPU 數量調整 batch size
        print(f"使用 {len(gpu_ids)} 個 GPU: {gpu_ids}")
        print(f"Batch size 調整為: {BATCH_SIZE}")
    else:
        BATCH_SIZE = 32
        gpu_ids = None
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4 if opt.multi_gpu else 0,  # 多 GPU 時使用多進程
        collate_fn=collate_fn,
        pin_memory=True  # 加速 CPU to GPU 傳輸
    )
    print('Dataloader is created')

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"主設備: {device}")
    
    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    vocab_size = len(event2word)
    print(f"詞彙表大小: {vocab_size}")
    
    start_epoch = 1
    model = TransformerModel(vocab_size=vocab_size)
    
    # 多 GPU 包裝
    if opt.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"✅ 啟用 DataParallel,使用 GPU: {gpu_ids}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    print('Model is created \nStart training')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {total_params:,}")
    if opt.multi_gpu and isinstance(model, nn.DataParallel):
        print(f"每個 GPU 的有效 batch size: {BATCH_SIZE // len(gpu_ids)}")
    
    model.train()
    losses = []
    best_loss = float('inf')

    os.makedirs(opt.ckp_folder, exist_ok=True)
    
    for epoch in range(start_epoch, epochs+1):
        single_epoch = []
        for x, y in tqdm(train_dataloader, desc=f'Epoch {epoch}'):
            x = x.to(device)
            y = y.to(device)
            
            min_len = min(x.size(1), y.size(1))
            x = x[:, :min_len]
            y = y[:, :min_len]
            
            output_logit = model(x)
            loss = nn.CrossEntropyLoss()(output_logit.permute(0,2,1), y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            single_epoch.append(loss.to('cpu').mean().item())
            optimizer.step()
            optimizer.zero_grad()
        
        single_epoch = np.array(single_epoch)
        losses.append(single_epoch.mean())
        print(f'>>> Epoch: {epoch}, Loss: {losses[-1]:.5f}')
        
        should_save = (epoch % SAVE_INTERVAL == 0) or (epoch == epochs)
        
        if should_save:
            checkpoint_path = os.path.join(opt.ckp_folder, 'epoch_%03d.pkl' % epoch)
            # 保存時處理 DataParallel
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'loss': losses[-1],
                'vocab_size': vocab_size,
            }, checkpoint_path)
            print(f'>>> Checkpoint saved: {checkpoint_path}')
        
        np.save(os.path.join(opt.ckp_folder, 'training_loss'), np.array(losses))
        
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_model_path = os.path.join(opt.ckp_folder, 'best_model.pkl')
            # 保存時處理 DataParallel
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'loss': losses[-1],
                'vocab_size': vocab_size,
            }, best_model_path)
            print(f'>>> Best model updated: {best_model_path} (Loss: {losses[-1]:.5f})')

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
