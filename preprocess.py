from collections import defaultdict
import csv
import pdb
import json
import random

def split_corpus2model(path):
    instances = []
    len2num = defaultdict(int)
    csv_reader = csv.reader(open(path))
    count = 0
    for line in csv_reader:
        if count == 0:
            pass
        else:
            instances.append(line)
            len2num[len(line[1])] += 1
        count += 1
    
    statistic_len(len2num, instances)
    random.shuffle(instances)
    
    valid_num = int(len(instances)*0.1)
    test_num = int(len(instances)*0.1)
    
    valid_data = instances[:valid_num]
    test_data = instances[valid_num:valid_num+test_num]
    train_data = instances[valid_num+test_num:]
    
    write2file(valid_data, path+'valid.json')
    write2file(test_data, path+'test.json')
    write2file(train_data, path+'train.json')


def write2file(data, path):
    # 数据写入文本
    file_result = open(path, 'w', encoding='utf-8')
    for line in data:
        file_result.write(json.dumps({'text':line[1],'label':line[0]}, ensure_ascii=False) + '\n')

def statistic_len(len2num, instances):
    # 数据长度分布统计
    sort_dict = sorted(len2num.items(), key= lambda x:x[0])
    probs = 0
    for data in sort_dict:
        len_ = data[0]
        num = data[1]
        probs += num/len(instances)
        print('文本长度小于{0}的数据占比为{1}'.format(len_, probs))
    
    


if __name__ == '__main__':
    path = './resources/raw_corpus/ChnSentiCorp_htl_all.csv'
    split_corpus2model(path)