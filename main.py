import torch
import pickle as pkl
import os
import argparse
from config import Config
import json
from train import Trainer

def main(args, cfg_path):
    
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    print('Device {}'.format(device))

    if (args.cont and args.mode == 'train') or args.mode == 'test':
        config = Config(os.path.join('./model', args.name, args.name + '.json'))
    else:
        config = Config(cfg_path)
    
    trainer = Trainer(config, device, args.mode, args.cont)

    if not os.path.exists(os.path.join('./model', config.model_name)):
        os.makedirs(os.path.join('./model', config.model_name))

    if not os.path.exists(os.path.join('./loss', config.model_name)):
            os.makedirs(os.path.join('./loss', config.model_name))

    if args.mode == 'train':
        # Save Configuration
        with open(os.path.join('./model', config.model_name, config.model_name + '.json'), 'w') as out:
            json.dump(config.dict, out)
        
        # Train
        loss_info = trainer.train()
        with open(os.path.join('./loss', config.model_name, config.model_name + '.pkl'), 'wb') as out:
            pkl.dump(loss_info, out)
        

    else:
        if not os.path.exists(os.path.join('./result', config.model_name)):
            os.makedirs(os.path.join('./result', config.model_name))

        trainer.test()
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('-c', '--cont', action='store_true', default=None)
    parser.add_argument('-n', '--name', type=str, required=False)
    args = parser.parse_args()

    main(args, './config.json')
    # sena_main(args, './config.json')