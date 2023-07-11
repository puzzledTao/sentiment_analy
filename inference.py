import sys
import pdb
import json
import torch
import logging
import configparser
from optparse import OptionParser
from corecode.nn.models import bertModel
from corecode.data.dataset import DataIter
from corecode.infer.inference import Inference
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
    path_test = config['PATH']['test']
    max_len = int(config['DATA']['max_len'])
    batch_size = int(config['DATA']['batch_size'])
    
    config_dict = {
        'path':path_test,
        'max_len': max_len,
        'batch_size': batch_size,
    }
    
    test_data = DataIter(config_dict, is_shuffle=False, is_inference=False)
    
    logger.info('test data size: {0}'.format(test_data.data_count))
    
    return test_data

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

def init_infer(config, model, test_data, logger, device):
    model_infer = config['PATH']['model_infer']
    max_len = int(config['DATA']['max_len'])
    result = config['PATH']['result']
    
    config_dict = {
        'model': model, 'test_data': test_data,'logger': logger,
        'model_infer': model_infer,'result': result, 'device':device
    }
    
    infer = Inference(config=config_dict)
    
    return infer
    

def run_inference():
    opts = init_opts()
    config = configparser.ConfigParser()
    config.read(opts.config, encoding='utf-8')
    logger = log_info(config['PATH']['log'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('init dataset...')
    
    test_data = init_data_loader(config, logger)
    logger.info('init model...')
    model = init_model(device, config=config)
    logger.info('init infer...')
    infer = init_infer(config, model, test_data, logger, device)
    logger.info('start infer...')
    infer.infer()


if __name__ == '__main__':
    run_inference()