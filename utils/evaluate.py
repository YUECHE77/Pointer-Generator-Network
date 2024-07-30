import torch
import lawrouge

from .PGDataset import origin_sent2id, source2id


def encode(sents, tokenizer, max_length):
    source_sent_ext, oovs, source_len = source2id(sents, tokenizer, length=max_length,
                                                  vocab_size=len(tokenizer.word2idx))

    max_oov_num = len(oovs)
    source_sent = origin_sent2id(sents, tokenizer, length=max_length)

    # 注意，这里返回的每一个元素都被放到了一个list里，这是因为要给这些值加上batch这个维度，否则无法输入到模型中
    return torch.tensor([source_sent]), torch.tensor([source_sent_ext]), torch.tensor([max_oov_num]), torch.tensor([source_len]), oovs


def greedy_search(sents, tokenizer, encoder, decoder, reduce_state, num_steps, max_length=512, hidden_dim=256):
    vocab_size = len(tokenizer.word2idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)
    reduce_state.to(device)

    encoder.eval()
    decoder.eval()
    reduce_state.eval()

    source, source_ext, oov_num, source_length, oov_dict = encode(sents, tokenizer, max_length=max_length)
    source, source_ext, oov_num = source.to(device), source_ext.to(device), oov_num.to(device)

    decode_words = []
    decode_ids = tokenizer.word2idx['<s>']

    with torch.no_grad():
        source_ext = source_ext[:, :torch.max(source_length).item()]
        coverage = torch.zeros((1, torch.max(source_length).item()), device=device)
        source_mask = (source_ext != 0).float().to(device)

        #         print('source_ext shape: ', source_ext.shape)
        #         print('coverage shape: ', coverage.shape)
        #         print('source_mask shape: ', source_mask.shape)
        #         print()

        c_t_1 = torch.zeros((1, 2 * hidden_dim), device=device)
        #         print('c_t_1 shape', c_t_1.shape)

        encoder_outputs, encoder_feature, encoder_hidden = encoder(source, source_length)
        s_t_1 = reduce_state(encoder_hidden)

        #         print('encoder_outputs shape: ', encoder_outputs.shape)
        #         print('encoder_feature shape: ', encoder_feature.shape)
        #         print('s_t_1 is a tuple with length: ', len(s_t_1))
        #         print()

        for di in range(num_steps):
            y_t_1 = torch.tensor([decode_ids], device=device)  # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
            max_art_oovs = torch.max(oov_num)
            extra_zeros = torch.zeros((1, max_art_oovs), device=device)

            #             print('y_t_1 shape: ', y_t_1.shape)
            #             print('extra_zeros shape: ', extra_zeros.shape)

            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = decoder(y_t_1, s_t_1, encoder_outputs,
                                                                                encoder_feature, source_mask, c_t_1,
                                                                                extra_zeros, source_ext, coverage, di)

            num, index = torch.max(final_dist, 1)
            index = index.item()

            if index < vocab_size:
                word = tokenizer.idx2word[index]
                decode_ids = index

                if word == '</s>':
                    break

                decode_words.append(word)
            else:
                decode_words.append(oov_dict[index - vocab_size])
                decode_ids = tokenizer.word2idx['<OOV>']

            coverage = next_coverage

    result = ''.join(decode_words).replace('<OOV>', '')

    return result.strip().replace(' ', '')


def compute_rouge(prediction, target):
    """
    lawrouge可以直接处理一整个batch的数据。故这里的prediction和target都是list，每个元素是一个预测值或真实值(这俩都是文字，不是tokens)
    """
    prediction = [pred.strip().replace(' ', '') for pred in prediction]  # 中文不应该有空格
    target = [tgt.strip().replace(' ', '') for tgt in target]  # 中文不应该有空格

    rouge = lawrouge.Rouge()
    result = rouge.get_scores(prediction, target, avg=True)

    result = {
        'rouge-1': result['rouge-1']['f'],
        'rouge-2': result['rouge-2']['f'],
        'rouge-l': result['rouge-l']['f']
    }

    result = {key: value * 100 for key, value in result.items()}

    return result
