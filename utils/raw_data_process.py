import json
import random


def get_data(source_path, target_path, output_file, use_partial=None):
    """
    Used for processing LCSTC dataset

    :param source_path: train(test or valid).src.txt
    :param target_path: train(test or valid).tgt.txt
    :param output_file: The path of final json file you want to store
    :param use_partial: Number of datapoints you want to use
    :return: None
    """
    with open(source_path, 'r', encoding='utf-8') as src_f, open(target_path, 'r', encoding='utf-8') as tar_f:
        src_line = src_f.readlines()
        tar_line = tar_f.readlines()

    assert len(src_line) == len(tar_line), 'Source and target files must have the same number of datapoints'

    if use_partial is not None:
        src_line = src_line[:use_partial]
        tar_line = tar_line[:use_partial]

    all_data = []
    for src, tar in zip(src_line, tar_line):
        data = dict()
        data['source'] = src.strip()
        data['target'] = tar.strip()

        all_data.append(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_data:
            json_sample = json.dumps(sample, ensure_ascii=False)
            f.write(json_sample)
            f.write('\n')


def get_data_nlpcc(path, train_output_path, test_output_path, use_partial=None, test_size=1000):
    """
    Used for nlpcc dataset -> doesn't have test/val test

    :param path: nlpcc2017_clean.json
    :param train_output_path: The path of final json file (Training data) you want to store
    :param test_output_path: The path of final json file (Test data) you want to store
    :param use_partial: Number of Training datapoints you want to use
    :param test_size: Number of Testing datapoints you want to use
    :return: None
    """
    with open(path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    all_data = all_data['data']  # a list
    random.shuffle(all_data)

    test_data = all_data[:test_size]
    train_data = all_data[test_size:]

    print(f'train data size: {len(train_data)}')
    print(f'test data size: {len(test_data)}')

    if use_partial is not None:
        train_data = train_data[:use_partial]

    with open(train_output_path, 'w', encoding='utf-8') as f:
        for sample in train_data:
            json_sample = json.dumps(sample, ensure_ascii=False)
            f.write(json_sample)
            f.write('\n')

    with open(test_output_path, 'w', encoding='utf-8') as f:
        for sample in test_data:
            json_sample = json.dumps(sample, ensure_ascii=False)
            f.write(json_sample)
            f.write('\n')


if __name__ == '__main__':
    # if you use nlpcc dataset:

    get_data_nlpcc('../dataset/nlpcc/nlpcc2017_clean.json', '../dataset/nlpcc/nlpcc_train.json',
                   '../dataset/nlpcc/nlpcc_test.json', test_size=2000)

    # if you use LCSTS dataset:

    # get_data('../dataset/LCSTS/train.src.txt', '../dataset/LCSTS/train.tgt.txt',
    #          '../dataset/LCSTS/train_data.json', use_partial=50000)
    #
    # get_data('../dataset/LCSTS/test.src.txt', '../dataset/LCSTS/test.tgt.txt',
    #          '../dataset/LCSTS/test_data.json')
    #
    # get_data('../dataset/LCSTS/valid.src.txt', '../dataset/LCSTS/valid.tgt.txt',
    #          '../dataset/LCSTS/val_data.json')
