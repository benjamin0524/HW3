import torch
import pickle
import numpy as np
import os
import argparse
import utils
import torch.nn as nn
import math

X_LEN = 512
EPOCHS = 50 
SAVE_INTERVAL = 10  

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path', type=str, help='The dictionary path', default='./dictionary.pkl')
    parser.add_argument('--device', type=str, help='Device to run the model on', default='cuda')
    parser.add_argument('--model_path', type=str, help='Path to the trained model checkpoint', required=True)
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt MIDI file (8 bars)', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save the generated continuation MIDI', required=True)
    parser.add_argument('--n_target_bar', type=int, help='Number of bars to generate', default=24)
    parser.add_argument('--temperature', type=float, help='Sampling temperature', default=1.2)
    parser.add_argument('--topk', type=int, help='Top-k for sampling', default=5)
    args = parser.parse_args()
    return args

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

        # create causal mask (to prevent looking ahead)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)

        x = self.transformer(x, mask=mask)
        output = self.fc_out(x)

        return output

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

def test_continuation(args, prompt_path, n_target_bar=24, temperature=1.2, topk=5, output_path='', model_path=''):
    """
    Continuation generation for Task 2
    """
    event2word, word2event = pickle.load(open(args.dict_path, 'rb'))
    device = args.device

    note_items, tempo_items = utils.read_items(prompt_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end

    items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)

    words = []
    for event in events:
        e = '{}_{}'.format(event.name, event.value)
        if e in event2word:
            words.append(event2word[e])
        else:
            if event.name == 'Note Velocity':
                if 'Note Velocity_21' in event2word:
                    words.append(event2word['Note Velocity_21'])
                else:
                    velocity_keys = [k for k in event2word.keys() if 'Note Velocity' in k]
                    if velocity_keys:
                        words.append(event2word[velocity_keys[0]])

    prompt_bars = words.count(event2word['Bar_None'])
    print(f"Prompt contains {prompt_bars} bars")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab_size = checkpoint.get('vocab_size', len(event2word))
    model = TransformerModel(vocab_size=vocab_size).to(device)

    state_dict = checkpoint['model']
    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    current_generated_bar = 0
    max_tokens = n_target_bar * 500  # maximum token limit
    token_count = 0

    print(f'Start generating continuation for {n_target_bar} bars...')
    while current_generated_bar < n_target_bar and token_count < max_tokens:
        token_count += 1
        if token_count % 100 == 0:
            print(f"\rGenerated {current_generated_bar}/{n_target_bar} bars, tokens: {token_count}", end="")

        # Take the last X_LEN tokens as context
        temp_x = torch.tensor([words[-X_LEN:]], dtype=torch.long).to(device)
        output_logits = model(temp_x)

        _logit = output_logits[0, -1, :].cpu().detach()
        word = temperature_sampling(_logit, temperature, topk)

        words.append(word)

        if word == event2word['Bar_None']:
            current_generated_bar += 1

    print(f"\nGeneration completed with {len(words)} tokens")

    utils.write_midi(
        words=words,
        word2event=word2event,
        output_path=output_path,
        prompt_path=prompt_path
    )
    print(f"Saved generated MIDI to: {output_path}")

def main():
    args = parse_opt()

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    test_continuation(
        args=args,  # Pass the args as an argument
        prompt_path=args.prompt_path,
        n_target_bar=args.n_target_bar,
        temperature=args.temperature,
        topk=args.topk,
        output_path=args.output_path,
        model_path=args.model_path
    )

if __name__ == '__main__':
    main()
