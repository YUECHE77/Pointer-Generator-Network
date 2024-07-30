import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils.tokenizer import build_tokenizer
from utils.PGDataset import SumDataset
from utils.optimizer import AdagradCustom

from model.PGnet import Encoder, Decoder, ReduceState

if __name__ == '__main__':
    # ----------------------------------------------------#
    #   data_path:      Path to the training set
    #   src_name:       Column name of source
    #   tgt_name:       Column name of target
    # ----------------------------------------------------#
    data_path = 'dataset/nlpcc/nlpcc_test.json'
    src_name = 'content'
    tgt_name = 'title'
    # ----------------------------------------------------#
    #   Build your tokenizer, and load the dataset:

    #   n_src_vocab:        Size of your tokenizer(vocab)
    #   min_freq:           Min-Frequency of words
    #   existed_txt_path:   If you already have the txt file

    #   max_source_length:  Max length of source content
    #   max_target_length:  Max length of target content
    # ----------------------------------------------------#
    n_src_vocab = 40000
    min_freq = 3
    existed_txt_path = 'dict.txt'  # Set to None if you don't have the txt file

    max_source_length = 512
    max_target_length = 512

    tokenizer = build_tokenizer(data_path, src_name, n_src_vocab, min_freq, existed_txt_path)

    dataset = SumDataset(data_path, tokenizer, max_source_length=max_source_length, max_target_length=max_target_length,
                         src_name=src_name, tgt_name=tgt_name)
    # ----------------------------------------------------#
    #   Set the training arguments:

    #   batch_size:     Batch size
    #   epoch_num:      Number of epochs
    #   lr:             Learning rate

    #   emb_dim:        Embedding dimension
    #   hidden_dim:     Hidden dimension
    #   vocab_size:     Size (length) of you tokenizer (vocab)
    # ----------------------------------------------------#
    batch_size = 8
    epoch_num = 80
    lr = 0.15
    emb_dim = 128
    hidden_dim = 256
    vocab_size = len(tokenizer.word2idx)
    # ----------------------------------------------------#
    #   Initialize the networks (and dataloader)
    # ----------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nThe device you are using is {device}\n')

    encoder = Encoder(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim)
    decoder = Decoder(vocab_size=vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, pointer_gen=True)
    reduce_state = ReduceState(hidden_dim=hidden_dim)

    encoder.to(device)
    decoder.to(device)
    reduce_state.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # ----------------------------------------------------#
    #   Set up optimizer
    # ----------------------------------------------------#
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(reduce_state.parameters())
    optimizer = AdagradCustom(params, lr=lr, initial_accumulator_value=0.1)
    # ----------------------------------------------------#
    #   Start training!!!
    # ----------------------------------------------------#
    print('\nStart Training!!! \n')

    best_loss = 1e5

    for epoch in range(epoch_num):
        total_batches = len(dataloader)

        count = 0
        avg_loss = 0

        encoder.train()
        decoder.train()
        reduce_state.train()

        with tqdm(total=total_batches, desc=f'Epoch {epoch+1}/{epoch_num}', unit='batch') as pbar:

            for source, target, source_ext, target_ext, oov_num, source_length, target_length in dataloader:
                # source:        (batch_size, max_source_length) -> max_source_length = 512
                # target:        (batch_size, max_target_length) -> max_target_length = 512
                # source_ext:    (batch_size, max_source_length)
                # target_ext:    (batch_size, max_target_length)
                # oov_num:       (batch_size, )
                # source_length: (batch_size, )
                # target_length: (batch_size, )

                optimizer.zero_grad()

                if source.size(0) != batch_size:
                    continue

                source, target, source_ext, target_ext, oov_num = source.to(device), target.to(device), source_ext.to(device), target_ext.to(device), oov_num.to(device)

                bos = torch.tensor([tokenizer.word2idx['<s>']] * target_ext.shape[0], device=device).reshape((-1, 1))  # (batch_size, 1)
                target = torch.cat([bos, target], dim=1)

                max_src_len = torch.max(source_length).item()
                max_tgt_len = torch.max(target_length).item()

                source_attention_mask = (source_ext != 0).float().to(device)[:, :max_src_len]
                target_attention_mask = (target_ext != 0).float().to(device)[:, :max_tgt_len]

                source_ext = source_ext[:, :max_src_len]

                coverage = torch.zeros((batch_size, max_src_len), device=device)  # [batch_size, max(seq_lens)]
                c_t_1 = torch.zeros((batch_size, 2 * hidden_dim), device=device)  # [batch_size, 2 * hidden_dim]
                step_losses = []

                encoder_outputs, encoder_feature, encoder_hidden = encoder(source, source_length)

                s_t_1 = reduce_state(encoder_hidden)

                for di in range(max_tgt_len):
                    y_t_1 = target[:, di]  # 取一个时间步的target

                    max_art_oovs = torch.max(oov_num)
                    extra_zeros = torch.zeros((batch_size, max_art_oovs), device=device)  # (batch_size, max_oov_num)

                    final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = decoder(y_t_1, s_t_1,
                                                                                        encoder_outputs,
                                                                                        encoder_feature,
                                                                                        source_attention_mask,
                                                                                        c_t_1, extra_zeros, source_ext,
                                                                                        coverage, di)

                    target_word = target_ext[:, di]  # 摘要的下一个单词的编码 -> 即label

                    gold_probs = torch.gather(final_dist, 1, target_word.unsqueeze(1)).squeeze()  # 取出目标单词的概率gold_probs
                    step_loss = -torch.log(gold_probs + 1e-10)  # 最大化gold_probs，也就是最小化step_loss（添加负号）

                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                    step_loss = step_loss + step_coverage_loss

                    coverage = next_coverage

                    step_mask = target_attention_mask[:, di]
                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)

                sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
                avg_len = torch.sum(target_attention_mask, 1)
                sum_losses = sum_losses / avg_len

                loss = torch.mean(sum_losses)

                if torch.isnan(loss):
                    print("NaN loss detected. Skipping this batch.")
                    continue  # 如果损失是NaN，跳过这个batch

                loss.backward()

                count += 1
                avg_loss += loss.item()

                clip_grad_norm_(encoder.parameters(), 1.0)
                clip_grad_norm_(decoder.parameters(), 1.0)
                clip_grad_norm_(reduce_state.parameters(), 1.0)

                optimizer.step()

                torch.cuda.empty_cache()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

                # if count == 300:
                #     break

        print(f'Average loss in epoch {epoch + 1:02} is {avg_loss / count:.04}')

        # break

        avg_loss = round(avg_loss / count, 4)
        if avg_loss < best_loss:
            print("---- save with best loss ----", epoch + 1)
            best_loss = avg_loss
            # 保存模型
            torch.save(encoder.state_dict(), 'logs/encoder' + "_epoch=" + str(epoch + 1) + "_best_loss=" + str(best_loss) + ".pth")
            torch.save(decoder.state_dict(), 'logs/decoder' + "_epoch=" + str(epoch + 1) + "_best_loss=" + str(best_loss) + ".pth")
            torch.save(reduce_state.state_dict(), 'logs/reduce_state' + "_epoch=" + str(epoch + 1) + "_best_loss=" + str(best_loss) + ".pth")

    print('\nFinished Training!!! \n')
