import os
import glob
import torch
import pickle
import argparse
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
X_LEN = 512
EPOCHS = 200
SAVE_INTERVAL = 10
def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path', type=str, default='./dictionary.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckp_folder', type=str, default='./checkpoints')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--output_dir', type=str, default='./results')
    return parser.parse_args()

class MusicData(Dataset):
    def __init__(self, midi_files, dict_path):
        self.midi_files = midi_files
        self.x_len = X_LEN
        self.dict_path = dict_path
        self.event2word, self.word2event = pickle.load(open(self.dict_path, 'rb'))
        self.data = self.prepare_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def extract_events(self, path):
        note_items, tempo_items = utils.read_items(path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        return utils.item2event(groups)

    def prepare_data(self):
        all_events = []
        for midi in tqdm(self.midi_files, desc="Extracting events"):
            try:
                events = self.extract_events(midi)
                all_events.append(events)
            except Exception as e:
                print(f"Error processing {midi}: {e}")

        all_words = []
        for events in tqdm(all_events, desc="Converting to tokens"):
            words = [self.event2word.get(f'{event.name}_{event.value}', 
                                        self.event2word.get('Note Velocity_21', 
                                                           self.event2word.get('Note Velocity', 0))) 
                     for event in events]
            all_words.append(words)

        segments = []
        for words in all_words:
            step = self.x_len // 2
            for i in range(0, len(words) - self.x_len, step):
                segments.append([words[i:i+self.x_len], words[i+1:i+self.x_len+1]])

        return segments

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_head=12, n_layer=12, d_inner=2048):
        super(TransformerModel, self).__init__()

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=X_LEN,
            n_ctx=X_LEN,
            n_embd=d_model,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=d_inner,
            activation_function="gelu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=False,
        )
        self.transformer = GPT2LMHeadModel(config)

    def forward(self, x):
        return self.transformer(input_ids=x).logits

def train_model(continue_from_checkpoint=False, checkpoint_path=''):
    train_files = glob.glob('./Pop1K7/midi_analyzed/**/*.mid', recursive=True) or glob.glob('./Pop1K7/midi_analyzed/*.mid')
    print(f"Training files count: {len(train_files)}")

    train_dataset = MusicData(train_files, dict_path=opt.dict_path)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    vocab_size = len(event2word)

    model = TransformerModel(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    if continue_from_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1

    model.train()
    losses = []
    best_loss = float('inf')
    os.makedirs(opt.ckp_folder, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_losses = []
        for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            min_len = min(x.size(1), y.size(1))
            x, y = x[:, :min_len], y[:, :min_len]

            output = model(x)
            loss = nn.CrossEntropyLoss()(output.permute(0, 2, 1), y)
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.5f}")

        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            checkpoint_path = os.path.join(opt.ckp_folder, f"epoch_{epoch:03d}.pkl")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': vocab_size,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(opt.ckp_folder, "best_model.pkl")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': vocab_size,
            }, best_model_path)
            print(f"Best model saved: {best_model_path}")

def main():
    global opt
    opt = get_opts()

    if opt.mode == 'train':
        train_model()

if __name__ == '__main__':
    main()
