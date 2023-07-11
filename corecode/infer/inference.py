import torch

class Inference(object):
    def __init__(self, config):
        self._model = config['model']
        self._test_data = config['test_data']
        self._logger = config['logger']
        self._model_infer = config['model_infer']
        self._result = config['result']
        self._device =  config['device']
        
        # 加载模型参数
        self._model.load_state_dict(torch.load(self._model_infer))
    
    def infer(self):
        corr_num = 0
        all_num = 0
        instance_count = 0
        self._model.to(self._device)
        self._model.eval()
        for i, input_dict in enumerate(self._test_data.generate_data()):
            texts = input_dict['texts']
            labels = input_dict['labels']
            output = self._model(texts, labels)
            probs = output['probs']
            max_index = torch.argmax(probs, dim=-1)
            equal_compare = torch.eq(max_index, torch.tensor(labels).to(self._device)).int()
            equal_num = torch.sum(equal_compare)
            corr_num += equal_num.item()
            all_num += max_index.shape[0]
            
            instance_count += len(texts)
            self._logger.info('{0}/{1}'.format(instance_count, self._test_data.data_count))
        
        acc = corr_num/all_num
        self._logger.info('the acc is {0}'.format(acc))
        
        
        
        