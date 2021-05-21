import cv2
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import utils.loss as module_loss
import utils.metric as module_metric
# import model.model as module_arch
import model as module_arch
from parse_config import ConfigParser
from utils.gradcam import GradCam
from torchvision.utils import make_grid
from utils.plot import plot_tsne, plot_gram_cam, plot_lda
import numpy as np

from model.GMM import GaussianMixtureModel


def save_grid(im, im_name):
    im = make_grid(im, nrow=im.size(0)//4, normalize=True)
    npimg = im.numpy().transpose(1, 2, 0)*255
    # print(npimg.shape)
    cv2.imwrite('saved/imgs/gradcam/'+im_name, npimg)


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    old_bs = config['train_loader']['args']["batch_size"]
    tgt_cls = config['target_cls'] if config['target_cls'] > 0 else None
    data_loader = getattr(module_data, config['train_loader']['type'])(
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        mode='test',
        num_workers=0,
        N_sample=None,
        target_cls=tgt_cls,
        gray=config['train_loader']['args']["gray"]
    )

    # build model architecture, then print to console
    if 'Kernel' in config["name"]:
        model = config.init_obj('arch', module_arch,
                                N=old_bs)
    else:
        model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    grad_cam = GradCam(model)

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    gmm = GaussianMixtureModel(n_fea=128, n_class=10, device=device)

    for it, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            embeddings = model.embedding(data, all_ret=False)
            gmm.fit(embeddings, max_iter=100)
            output = gmm.predict(embeddings)
            pred_lab = torch.argmax(output, dim=1)
            # print(pred_lab.size())
            # exit()
            plot_tsne(embeddings.cpu().numpy(),
                      pred_lab.cpu().numpy(),
                      model='GMM+it'+str(it),
                      labels_gt=target.cpu().numpy())
            # plot_tsne(embeddings.cpu().numpy(),
            #           pred_lab.cpu().numpy(), model='GMM+it'+str(it))
            break
            # loss = loss_fn(output, target)
            # logstr = 'Iter: {} Loss: {:.6f}'.format(it, loss.item())

        # computing loss, metrics on test set
        # loss = loss_fn(output, target)
        batch_size = data.shape[0]
        total_loss += loss.item() * batch_size
        for i, metric in enumerate(metric_fns):
            met = metric(output, target).cpu()
            # logstr = logstr+" {}: {:.6f}".format(metric.__name__, met)
            total_metrics[i] += met * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
