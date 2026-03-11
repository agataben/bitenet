from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from sklearn.metrics import accuracy_score
from utils import AverageValueMeter
from torch import nn
import torch
import yaml
import os
from os.path import join


def train(model, loaders, config_path = None,
          lr = 0.01, momentum = 0.99, epochs = 10,
          exp_name = 'exp', logdir = 'logs', ckpt_dir = 'ckpt', ckpt_file = None):

    if config_path is not None:
        with open(config_path, 'r') as file:
            conf = yaml.safe_load(file)

        lr = conf['learning_rate']
        momentum = conf['momentum']
        epochs = conf['epochs']
        logger.debug(f'Training parameters: {lr}, {momentum}, {epochs}')

    optimizer = SGD(model.parameters(), lr = lr, momentum = momentum)
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
    else:
        start_epoch = 0

    if logdir is not None:
        writer = SummaryWriter(join(logdir, exp_name))
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok = True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    global_step = 0
    for e in range(start_epoch, start_epoch + epochs):
        for mode in ['train','test']:
            loss_meter.reset()
            acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode == 'train'):
                for i, batch in enumerate(loaders[mode]):
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    output = model(x)
                    loss = criterion(output,y)
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    batch_size = x.shape[0]
                    global_step += batch_size
                    acc = accuracy_score(y.to('cpu'), output.to('cpu').max(1)[1])
                    loss_meter.add(loss.item(), batch_size)
                    acc_meter.add(acc, batch_size)
                    if mode == 'train':
                        if logdir is not None:
                            writer.add_scalar('loss/train', loss_meter.value(), global_step = global_step)
                            writer.add_scalar('accuracy/train', acc_meter.value(), global_step = global_step)
            if logdir is not None:
                writer.add_scalar('loss/'+ mode, loss_meter.value(), global_step = global_step)
                writer.add_scalar('accuracy/'+ mode, acc_meter.value(), global_step = global_step)

        if ckpt_dir is not None:
            torch.save({
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'epoch': e + 1
            }, ckpt_dir + '/%s-%d.pth' % (exp_name, e + 1))

    return model

