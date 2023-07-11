import json
import pdb
import torch
import math
from torch.optim import Adam

class Trainer(object):
    def __init__(self, config):
        self._model = config['model']
        self._lr = config['lr']
        self._chek_freq = config['chek_freq']
        self._save_path = config['save_path']
        self._max_len = config['max_len']
        self._num_epochs = config['num_epochs']
        self._train_data = config['train_data']
        self._valid_data = config['valid_data']
        self._logger = config['logger']
        self._result = config['result']
        self._device = config['device']
        
        self.optimizer = Adam(self._model.parameters(), lr=self._lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = self.rule)
        self._model.train()
    
    def rule(self, epoch):
        lamda = math.pow(0.95, epoch)
        return lamda
    
    def fit(self):

        train_loss = 0
        batch_count = 1
        self._model.to(self._device)
        for epoch in range(self._num_epochs):
            self._logger.info('*****epoch={0}******'.format(epoch))
            for i, input_dict in enumerate(self._train_data.generate_data()):
                texts = input_dict['texts']
                labels = torch.tensor(input_dict['labels'])
                output = self._model(texts, labels)
                loss = output['loss']
                train_loss += loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                file_result = open(self._result, 'w', encoding='utf-8')
                if batch_count % self._chek_freq == 0:
                    train_loss /= self._chek_freq
                    self._model.eval()
                    corr_num = 0
                    all_num = 0
                    for i, input_dict in enumerate(self._valid_data.generate_data()):
                        texts = input_dict['texts']
                        labels = input_dict['labels']
                        output = self._model(texts, labels)
                        probs = output['probs']
                        max_index = torch.argmax(probs, dim=-1)
                        equal_compare = torch.eq(max_index, torch.tensor(labels).to(self._device)).int()
                        equal_num = torch.sum(equal_compare)
                        corr_num += equal_num.item()
                        all_num += max_index.shape[0]
                        
                        for k in range(len(input_dict['texts'])):
                            text = input_dict['texts'][k]
                            label = max_index[k].item()
                            file_result.write(json.dumps({'text':text,'label':label}, ensure_ascii=False) + '\n')
                    
                    acc = corr_num/all_num
                    self._logger.info('step={0}, loss={1}, all_num={2}, right_num={3}, acc={4}, lr={5}'.format(
                        batch_count, train_loss, all_num, corr_num, acc, 
                        self.optimizer.state_dict()['param_groups'][0]['lr']))
                    train_loss = 0
                    torch.save(self._model.state_dict(), self._save_path.format(acc))
                    self._model.train()
            
            self.scheduler.step()
                
                
        
        