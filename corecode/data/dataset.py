import pdb
import json
import random
class DataIter(object):
    def __init__(self, config, is_shuffle, is_inference):
        # 初始化迭代器
        self._path = config['path']
        self._max_len = config['max_len']
        self.batch_size = config['batch_size']
        self._is_shuffle = is_shuffle
        self._is_inference = is_inference
        
        self.instances, self.data_count = self.create_all_instances()
    
    def create_all_instances(self):
        # 获取所有的数据
        instance_count = 0
        instances = []
        with open(self._path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            instance_count += 1
            instances.append(json.loads(line.strip()))
        
        if self._is_shuffle:
            random.shuffle(instances)
        
        return instances, instance_count
    
    def generate_data(self):
        # 构造batch训练数据
        texts = []
        labels = []
        for i in range(len(self.instances)):
            single_inst = self.instances[i]
            text = single_inst['text']
            label = int(single_inst['label'])
            texts.append(text)
            labels.append(label)
            
            if len(texts) == self.batch_size or i == len(self.instances) - 1:
                yield {'texts':texts, 'labels':labels}
                texts = []
                labels = []
            else:
                pass
        