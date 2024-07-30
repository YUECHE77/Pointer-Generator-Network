import torch

from utils.tokenizer import build_tokenizer
from utils.evaluate import greedy_search

from model.PGnet import Encoder, Decoder, ReduceState

if __name__ == '__main__':
    # ----------------------------------------------------#
    #   prompt:     Your input sentence
    #   max_length: Max length of your input sentence
    #   num_steps:  How many steps you want to generate
    # ----------------------------------------------------#
    prompt = """泉州新闻网6月26日讯(记者陈健通讯员庄凌龙蔡庆铭)福建晋江一男子结婚时承诺给妻子买金手镯,却苦于囊中羞涩,起了贪念,竟到珠宝柜抢夺金手镯。
    26日上午,记者从晋江市公安局获悉,犯罪嫌疑人何某抵不过心中的愧疚和害怕,主动向晋江警方投案;目前,何某因涉嫌抢夺被刑事拘留。6月21日晚上9点40许,
    何某来到晋江一家购物超市的珠宝柜,称要买手镯。据珠宝店陈女士介绍说,“他说要给女朋友买手镯,我拿了一个10克多的给他,他说要大一点的,
    我就拿了两个克数在40克左右的大手镯。男子看了一下手镯称要带女朋友一起来买,就离开了柜台,走出超市。”十几分钟后何某又进来,让陈女士拿手镯给他拍个照,
    他要发给女朋友。由于公司没有给顾客拍照这一说,陈女士特意请示了经理,才拿出一只手镯让男子拍照。“他拍了照片,然后就一直在玩弄手机。”陈女士称,她以为男子是在发照片。
    “想不到不一会儿功夫何某竟抢走了手镯,跑到超市对面的一条巷子,骑上一辆摩托车扬长而去。”当地警方接警后,调取了柜台监控录像及周边监控录像,基本锁定犯罪嫌疑人。
    据民警介绍,何某拿走的手镯重41.91克,价值约1万多元。26日上午10时,犯罪嫌疑人何某已主动向警方投案。据了解,何某今年25岁,是漳州人,在当地一处工地上班,一个月有五六千的收入。
    何某交代说,事发当日他身上带着剩下的1000多元工资,准备给老婆买金手镯,那是两人结婚时他对妻子的承诺,然而,囊中羞涩的他连最便宜的小手镯也买不起,一时贪念起便抢走手镯。
    “抢完后我也很害怕,有想要还给人家,怕遭到对方责怪。”何某说,手镯抢完后,他不敢给老婆,一直放在自己身上,每天晚上睡不着,直到25日才向父母袒露事情的经过,并下定决心投案。
    目前,案件正在进一步调查中。"""

    max_length = 512
    num_steps = 128
    # ----------------------------------------------------#
    #   Set the tokenizer
    #   See train.py for parameters' detail
    # ----------------------------------------------------#
    data_path = 'dataset/nlpcc/nlpcc_test.json'
    src_name = 'content'
    n_src_vocab = 40000
    min_freq = 2
    existed_txt_path = 'dict.txt'  # Set to None if you don't have the txt file

    tokenizer = build_tokenizer(data_path, src_name, n_src_vocab, min_freq, existed_txt_path)
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
    #   Start inference
    # ----------------------------------------------------#
    result = greedy_search(prompt, tokenizer, encoder, decoder, reduce_state, num_steps=num_steps,
                           max_length=max_length, hidden_dim=hidden_dim)

    print()
    print(result)
