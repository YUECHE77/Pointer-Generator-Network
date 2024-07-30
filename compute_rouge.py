import json
import torch

from utils.evaluate import greedy_search, compute_rouge
from utils.tokenizer import build_tokenizer

from model.PGnet import Encoder, Decoder, ReduceState

# ----------------------------------------------------#
#   test_path:  Path to your test set (or any of your dataset you want to compute rouge)
#   src_name:   Column name of source
#   tgt_name:   Column name of target
# ----------------------------------------------------#
test_path = 'dataset/nlpcc/nlpcc_test.json'
src_name = 'content'
tgt_name = 'title'
predictions = []
targets = []
# ----------------------------------------------------#
#   Set the tokenizer
#   See train.py for parameters' detail
# ----------------------------------------------------#
train_data_path = 'dataset/nlpcc/nlpcc_test.json'  # the dataset you use to initialize tokenizer
n_src_vocab = 40000
min_freq = 2
existed_txt_path = 'dict.txt'  # Set to None if you don't have the txt file

tokenizer = build_tokenizer(train_data_path, src_name, n_src_vocab, min_freq, existed_txt_path)
# ----------------------------------------------------#
#   encoder_model_path:      Path to the trained encoder model
#   decoder_model_path:      Path to the trained decoder model
#   reduce_state_model_path: Path to the trained ReduceState model
# ----------------------------------------------------#
encoder_model_path = 'logs/colab/encoder_epoch=30_best_loss=10000.pth'
decoder_model_path = 'logs/colab/decoder_epoch=30_best_loss=10000.pth'
reduce_state_model_path = 'logs/colab/reduce_state_epoch=30_best_loss=10000.pth'
# ----------------------------------------------------#
#   Parameters must be the same as you trained your model
# ----------------------------------------------------#
emb_dim = 128
hidden_dim = 256
vocab_size = len(tokenizer.word2idx)
# ----------------------------------------------------#
#   Load models
# ----------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The device you are using is {device}')

encoder = Encoder(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
decoder = Decoder(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, pointer_gen=True).to(device)
reduce_state = ReduceState(hidden_dim=hidden_dim).to(device)

encoder.load_state_dict(torch.load(encoder_model_path, map_location=device))
decoder.load_state_dict(torch.load(decoder_model_path, map_location=device))
reduce_state.load_state_dict(torch.load(reduce_state_model_path, map_location=device))

print('\nModels are successfully loaded!!!\n')
# ----------------------------------------------------#
#   Start to compute rouge
# ----------------------------------------------------#
with open(test_path, 'r', encoding='utf-8') as f:
    all_data = f.readlines()

    for line in all_data:
        dic = json.loads(line)
        source = dic[src_name]
        summary = dic[tgt_name]

        pred = greedy_search(source, tokenizer, encoder, decoder, reduce_state, num_steps=128,
                             max_length=512, hidden_dim=256)

        if not pred:
            continue

        targets.append(summary)
        predictions.append(pred)

    assert len(targets) == len(predictions), 'Targets and predictions should have same length'

    print()
    print(f'There are {len(predictions)} samples in test set')

rouge_score = compute_rouge(predictions, targets)

print()
print(rouge_score)
