import torch
import numpy as np
from sklearn.metrics import accuracy_score

# Setup logging
import logging
logger = logging.getLogger(__name__)


def test(model,test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    predictions, labels = [], []
    for batch in test_loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        output = model(x)
        preds = output.to('cpu').max(1)[1].numpy()
        labs = y.to('cpu').numpy()
        predictions.extend(list(preds))
        labels.extend(list(labs))

    logger.info(f'Accuracy: {accuracy_score(labels,predictions)}')

    return np.array(predictions), np.array(labels)

