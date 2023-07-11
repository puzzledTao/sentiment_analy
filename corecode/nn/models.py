import torch
import pdb
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class bertModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ptm_model = config['ptm_model']
        self.num_label = config['num_label']
        self.max_len = config['max_len']
        self.device = config['device']
        self.tokenizer = BertTokenizer.from_pretrained(self.ptm_model)
        self.model = BertModel.from_pretrained(self.ptm_model)
        self.dense = nn.Linear(768, self.num_label)
        
        
    def forward(self, texts, labels):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.device)
        outputs = self.model(**inputs)
        # (bs, max_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        # 获取cls的表示 (bs, hidden_size)
        cls_states = hidden_states[:,0,:]
        # (bs, num_label)
        cls_logits = self.dense(cls_states)
        probs = F.softmax(cls_logits, dim=1)
        log_probs = F.log_softmax(cls_logits, dim=1)
        # (bs, num_label)
        label_onehot = F.one_hot(torch.tensor(labels).to(self.device), num_classes=self.num_label)
        # (bs, num_label)
        loss = -log_probs*label_onehot
        # (bs,)
        loss = torch.sum(loss, axis=-1)
        # 标量
        loss = torch.mean(loss)
        
        output = {
            'loss':loss,
            'probs':probs
        }
        
        return output