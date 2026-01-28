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


# Setup logging
import logging
logger = logging.getLogger(__name__)

def train(model, loaders, lr = 0.01, 
          momentum = 0.99, epochs = 10,
          weight_dir = 'weights', exp_name = 'experiment',
          logdir = 'logs', config = False, ckpt_path = None):

    if config:
        with open('config.yml', 'r') as file:
            conf = yaml.safe_load(file)

        lr = conf['learning_rate']
        momentum = conf['momentum']
        epochs = conf['epochs']
        logger.debug(f'Training parameters: {lr}, {momentum}, {epochs}')

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer = SGD(model.parameters(), lr = lr, momentum = momentum)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
    else:
        optimizer = SGD(model.parameters(), lr = lr, momentum = momentum)
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    writer = SummaryWriter(join(logdir, exp_name))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    os.makedirs(weight_dir,exist_ok = True)
    global_step = 0
    for e in range(start_epoch, start_epoch + epochs):
        logger.info(f'Epoch {e + 1} of {epochs}')
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
                        writer.add_scalar('loss/train', loss_meter.value(), global_step = global_step)
                        writer.add_scalar('accuracy/train', acc_meter.value(), global_step = global_step)

            writer.add_scalar('loss/'+ mode, loss_meter.value(), global_step = global_step)
            writer.add_scalar('accuracy/'+ mode, acc_meter.value(), global_step = global_step)

        torch.save({
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, ckpt_path + '/%s-%d.pth' % (exp_name, e + 1))

        torch.save(model.state_dict(), weight_dir + '/%s-%d.pth' % (exp_name, e + 1))
    return model

