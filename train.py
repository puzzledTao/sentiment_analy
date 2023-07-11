import sys
import pdb
import json
import torch
import logging
import configparser
from optparse import OptionParser
from corecode.nn.models import bertModel
from corecode.train.trainer import Trainer
from corecode.data.dataset import DataIter
from transformers import BertTokenizer, BertModel


def init_opts():
    op = OptionParser()
    op.add_option(
        '-c', '--config', dest='config', type='str',
        help='path of configuration file'
    )
    argv = []
    if hasattr(sys.modules['__main__'], '__file__'):
        argv = sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if not opts.config:
        op.print_help()
        exit()

    return opts

def log_info(log_path):
    # 日志文件
    logger = logging.getLogger()
    # Log等级总开关
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关

    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def init_data_loader(config, logger):
    path_train = config['PATH']['train']
    path_valid = config['PATH']['valid']
    max_len = int(config['DATA']['max_len'])
    batch_size = int(config['DATA']['batch_size'])
    
    config_dict = {
        'path':path_train,
        'max_len': max_len,
        'batch_size': batch_size,
    }
    
    train_data = DataIter(config_dict, is_shuffle=True, is_inference=False)
    
    config_dict['path'] = path_valid
    valid_data = DataIter(config_dict, is_shuffle=False, is_inference=False)
    
    logger.info('batch_size: {0}'.format(train_data.batch_size))
    logger.info('train data size: {0}'.format(train_data.data_count))
    logger.info('valid data size: {0}'.format(valid_data.data_count))
    
    return train_data, valid_data
    
 
def init_model(device, config):
    ptm_model = config['PATH']['ptm_model']
    num_label = int(config['DATA']['num_label'])
    max_len = int(config['DATA']['max_len'])
    config_dict = {
        'ptm_model':ptm_model,
        'num_label':num_label,
        'max_len':max_len,
        'device':device
    }
    model = bertModel(config=config_dict)
    
    return model


def init_trainer(config, model, train_data, valid_data, logger, device):
    lr = float(config['MODEL']['learning_rate'])
    chek_freq = int(config['MODEL']['chek_freq'])
    save_path = config['PATH']['save_path']
    max_len = int(config['DATA']['max_len'])
    num_epochs = int(config['MODEL']['num_epochs'])
    result = config['PATH']['result']
    
    logger.info('learning rate={0}'.format(lr))
    logger.info('chek_freq={0}'.format(chek_freq))
    
    config_dict = {
        'model':model, 'lr':lr, 'chek_freq':chek_freq, 'save_path':save_path, 
        'max_len':max_len, 'num_epochs':num_epochs, 'train_data':train_data,
        'valid_data':valid_data, 'logger':logger, 'result':result,'device':device
    }
    trainer = Trainer(config=config_dict)
    
    return trainer
    
def run_train():
    opts = init_opts()
    config = configparser.ConfigParser()
    config.read(opts.config, encoding='utf-8')
    logger = log_info(config['PATH']['log'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('init dataset...')
    train_data, valid_data = init_data_loader(config=config, logger=logger)
    logger.info('init model...')
    model = init_model(device, config=config)
    logger.info('init train...')
    
    trainer = init_trainer(config, model, train_data, valid_data, logger, device)
    logger.info('start train...')
    trainer.fit()

    

if __name__ == '__main__':
    run_train()

