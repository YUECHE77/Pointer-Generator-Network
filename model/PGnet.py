import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_lstm_wt(lstm):
    """Initialize LSTM layer"""

    # lstm._all_weights -> [['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'],
    #                      ['weight_ih_l0_reverse', 'weight_hh_l0_reverse', 'bias_ih_l0_reverse', 'bias_hh_l0_reverse']]
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-0.02, 0.02)

            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    """Initialize linear layer"""
    linear.weight.data.normal_(std=1e-4)
    if linear.bias is not None:
        linear.bias.data.normal_(std=1e-4)


def init_wt_normal(wt):
    """Initialize embedding layer"""
    wt.data.normal_(std=1e-4)


def init_wt_unif(wt):
    wt.data.uniform_(-0.02, 0.02)


class Encoder(nn.Module):
    def __init__(self, vocab_size=50000, emb_dim=64, hidden_dim=128):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)

    # seq_lens: 1D tensor -> better be sorted descending
    def forward(self, input_x, seq_lens):
        """
        :param input_x:     source -> [batch_size, 512] 输入时确实是512。但输出时encoder_outputs的第二维度会变成 max(seq_lens)
        :param seq_lens:    seq_lens -> [batch_size, ] 这个list中的最大值就是 max(seq_lens)

        Example:
            X = torch.randint(1, 100, (4, 512))  # 0~100是随便设置的，实际上只要比tokenizer的 id2word 长度小即可
            source_len = torch.tensor([111, 222, 333, 444])

            此时输出的 encoder_outputs 的维度是 [4, 444, 256] -> [batch_size, max(seq_lens), 2 * hid_dim]
        """
        embedded = self.embedding(input_x)  # (batch_size, max(seq_lens), emb_dim)

        # 将填充过的序列打包，以便传入RNN/LSTM/GRU层时，忽略填充部分
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)

        output, hidden = self.lstm(packed)  # hidden is tuple([2, batch_size, hid_dim], [2, batch_size, hid_dim])

        # 将LSTM的输出恢复成填充的序列形式
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # [batch_size, max(seq_lens), 2 * hid_dim]

        encoder_outputs = encoder_outputs.contiguous()  # [batch_size, max(seq_lens), 2 * hid_dim]

        encoder_feature = encoder_outputs.view(-1, 2 * self.hidden_dim)
        encoder_feature = self.W_h(encoder_feature)  # [batch_size * max(seq_lens), 2 * hid_dim]

        # encoder_outputs: [batch_size, max(seq_lens), 2 * hid_dim]
        # encoder_feature: [batch_size * max(seq_lens), 2 * hid_dim]
        # hidden: tuple([2, batch_size, hid_dim], [2, batch_size, hid_dim])
        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self, hidden_dim=128):
        super(ReduceState, self).__init__()

        self.hidden_dim = hidden_dim

        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
        init_linear_wt(self.reduce_h)

        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = [2, batch_size, hidden_dim]

        # [2, batch_size, hidden_dim] -> [batch_size, 2, hidden_dim] -> [batch_size, hidden_dim * 2]
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))  # [batch_size, hidden_dim]

        # [2, batch_size, hidden_dim] -> [batch_size, 2, hidden_dim] -> [batch_size, hidden_dim * 2]
        c_in = c.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))  # [batch_size, hidden_dim]

        return hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)  # h, c: dim = [1, batch_size, hidden_dim]


class Attention(nn.Module):
    def __init__(self, hidden_dim, is_coverage):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_coverage = is_coverage

        if is_coverage:
            self.W_c = nn.Linear(1, hidden_dim * 2, bias=False)

        self.decode_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # Just a linear transformation

        # compute attention score
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        """
        加性注意力机制：

        :param s_t_hat:             results of concatenation of ReduceState -> [batch_size, 2 * hidden_dim] -> query
        :param encoder_outputs:     encoder output -> [batch_size, max(seq_lens), 2 * hid_dim] -> key and value
        :param encoder_feature:     reshaped encoder output -> [batch_size * max(seq_lens), 2 * hid_dim] -> encoder_outputs的reshape结果罢了，为了方便计算

        :param enc_padding_mask:    source attention mask -> [batch_size, max(seq_lens)]
        :param coverage:            coverage -> [batch_size, max(seq_lens)]
        """
        b, t_k, n = list(encoder_outputs.size())  # batch_size, max(seq_lens), 2 * hidden_dim

        dec_fea = self.decode_proj(s_t_hat)  # [batch_size, 2 * hidden_dim]

        # [batch_size, max(seq_lens), 2 * hidden_dim] -> 扩展解码器隐藏状态以匹配编码器输出的维度
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # [batch_size * max(seq_lens), 2 * hidden_dim] -> 调整维度以进行加法操作

        att_features = encoder_feature + dec_fea_expanded  # [batch_size * max(seq_lens), 2 * hidden_dim] -> 将编码器输出特征和解码器隐藏状态特征相加

        if self.is_coverage:
            coverage_input = coverage.view(-1, 1)  # [batch_size * max(seq_lens), 1] -> 将Coverage向量调整为合适的维度
            coverage_feature = self.W_c(coverage_input)  # [batch_size * max(seq_lens), 2*hidden_dim] -> 维度和att_features一致

            att_features = att_features + coverage_feature  # 将Coverage特征添加到注意力特征中

        e = torch.tanh(att_features)  # [batch_size * max(seq_lens), 2*hidden_dim] -> 通过tanh激活函数生成注意力能量

        scores = self.v(e)  # [batch_size * max(seq_lens), 1] -> 这就是注意力分数！！！
        scores = scores.view(-1, t_k)  # [batch_size, max(seq_lens)] -> 调整维度以匹配编码器输出的时间步长

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # [batch_size, max(seq_lens)] -> 计算注意力分布，并应用编码器输出的填充掩码
        # 就是你做softmax的后又有一些值乘以了0（enc_padding_mask）-> 因此此时的sun已经不为1了 -> 我们要重新计算此时的softmax结果
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor  # # 对注意力分布进行归一化

        attn_dist = attn_dist.unsqueeze(1)  # [batch_size, 1, max(seq_lens)] -> 即教材中的attention_weights -> [batch_size, num_query, num_key_value]

        c_t = torch.bmm(attn_dist, encoder_outputs)  # [batch_size, 1, 2 * hidden_dim] -> 计算上下文向量 -> 与value进行torch.bmm
        # 此时的c_t的维度就是[batch_size, num_query, value的维度]，下面又对其做了个reshape，其实没啥必要
        c_t = c_t.view(-1, self.hidden_dim * 2)  # [batch_size, 2*hidden_dim] -> 调整上下文向量的维度

        attn_dist = attn_dist.view(-1, t_k)  # [batch_size, max(seq_lens)] -> 调整注意力分布的维度

        if self.is_coverage:
            coverage = coverage.view(-1, t_k)  # [batch_size, max(seq_lens)]
            coverage = coverage + attn_dist  # 更新Coverage向量

        # c_t:          [batch_size, 2*hidden_dim]
        # attn_dist:    [batch_size, max(seq_lens)]
        # coverage:     [batch_size, max(seq_lens)]
        return c_t, attn_dist, coverage  # 返回上下文向量，注意力分布和更新后的Coverage向量


class Decoder(nn.Module):
    def __init__(self, vocab_size=50000, emb_dim=64, hidden_dim=128, pointer_gen=True, is_coverage=True):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.pointer_gen = pointer_gen

        self.attention_network = Attention(hidden_dim=hidden_dim, is_coverage=is_coverage)

        # decoder
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(hidden_dim * 2 + emb_dim, emb_dim)  # 用于将上下文向量和词嵌入拼接的线性层

        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if pointer_gen:
            self.p_gen_linear = nn.Linear(hidden_dim * 4 + emb_dim, 1)  # Pointer-Generator机制的线性层

        # p_vocab
        self.out1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out2 = nn.Linear(hidden_dim, vocab_size)  # 用于生成最终词汇分布的线性层
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        """
        Attention-Decoder:

        :param y_t_1:                   target[:, di] -> 第di个时间步的target -> [batch_size, ]
        :param s_t_1:                   result of ReduceState -> tuple([1, batch_size, hidden_dim], [1, batch_size, hidden_dim])
        :param encoder_outputs:         encoder output -> [batch_size, max(seq_lens), 2 * hid_dim]
        :param encoder_feature:         reshaped encoder output -> [batch_size * max(seq_lens), 2 * hid_dim]
        :param enc_padding_mask:        source attention mask -> [batch_size, max(seq_lens)]
        :param c_t_1:                   上下文向量 -> [batch_size, 2*hidden_dim]
        :param extra_zeros:             [batch_size, max_oov_num]
        :param enc_batch_extend_vocab:  source_ext -> [batch_size, max(seq_len)]
        :param coverage:                [batch_size, max(seq_lens)]
        :param step:                    di -> 一个数字罢了
        """

        # 即在预测模式的第一个时间步：
        if not self.training and step == 0:
            # s_t_1的结果是reduced state的结果 -> h_decoder, c_decoder: [1, batch, hidden_dim]
            h_decoder, c_decoder = s_t_1

            # torch.cat([batch_size, hidden_dim], [batch_size, hidden_dim], 1) -> [batch_size, 2*hidden_dim]
            s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim), c_decoder.view(-1, self.hidden_dim)), 1)

            # attention
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)  # [batch_size, emb_dim] -> y_t是一个时间步的target

        # torch.cat((c_t_1, y_t_1_embd), 1) -> torch.cat(([batch_size, 2*hidden_dim], [batch_size, emb_dim]), 1) -> [batch_size, 2*hidden_dim + emb_dim]
        # 最终x: [batch_size, emb_dim]
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))

        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)  # [batch_size, 1, hidden_dim], tuple([1, batch_size, hidden_dim], [1, batch_size, hidden_dim])

        h_decoder, c_decoder = s_t  # 均为[1, batch_size, hidden_dim]

        # torch.cat([batch_size, hidden_dim], [batch_size, hidden_dim], 1) -> [batch_size, 2*hidden_dim]
        s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim), c_decoder.view(-1, self.hidden_dim)), 1)

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        # 计算生成词汇的概率 p_gen，用于在词汇表生成和输入序列复制之间进行权衡
        p_gen = None
        if self.pointer_gen:
            # torch.cat(([batch_size, 2*hidden_dim], [batch_size, 2*hidden_dim], [batch_size, emb_dim]), 1)
            # -> [batch_size, 4*hidden_dim + emb_dim]
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)

            p_gen = self.p_gen_linear(p_gen_input)  # [batch_size, 1]
            p_gen = torch.sigmoid(p_gen)  # [batch_size, 1] -> 这就是所谓的“生成词汇表中词的概率” -> (1 - p_gen)就是“使用句中词汇的概率”

        # torch.cat(([batch_size, hidden_dim], [batch_size, 2*hidden_dim]), 1) -> [batch_size, 3*hidden_dim]
        output = torch.cat((lstm_out.view(-1, self.hidden_dim), c_t), 1)

        output = self.out1(output)  # [batch_size, hidden_dim]

        # output = F.relu(output)

        output = self.out2(output)  # [batch_size, vocab_size]
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:
            # 如果 pointer_gen 机制被启用，那么模型会同时使用词汇表中的词和输入序列中的词来生成最终的词汇分布

            # 生成词汇表中的词的概率:
            vocab_dist_ = p_gen * vocab_dist  # [batch_size, 1] * [batch_size, vocab_size] = [batch_size, vocab_size]

            # 生成输入序列中的词的概率:
            attn_dist_ = (1 - p_gen) * attn_dist  # [batch_size, 1] x [batch_size, max(seq_lens)] = [batch_size, max(seq_lens)]

            # 处理OOV词汇
            if extra_zeros is not None:
                # torch.cat(([batch_size, vocab_size], [batch_size, max_oov_num]), 1) -> [batch_size, vocab_size + max_oov_num]
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            # scatter_add 操作：对于enc_batch_extend_vocab的每一个索引i，将attn_dist_中对应位置的值添加到 vocab_dist_中的索引i位置
            # enc_batch_extend_vocab: [batch_size, max(seq_lens)]
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)

        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
