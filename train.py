from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from utils import *
from model import *
import bdcn
import cv2
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='/home/wanjin/dataset')
    parser.add_argument('--checkpoints_dir', type=str, default='./log')
    parser.add_argument('--model_name', type=str, default='MESFINet')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--load_pretrain_modelx2', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/MESFINet.pth.tar')
    parser.add_argument('--modelx2_path', type=str, default='./log/MESFINet_2xSR_final.pth.tar')
    parser.add_argument('--edge_model_path', type=str, default='./pretrained_moel/bdcn_pretrained_on_nyudv2_rgb.pth',
        help='the model to get the edge prior')
    return parser.parse_args()


def train(train_loader, cfg):
    device_ids=range(torch.cuda.device_count())
    cfg.cuda = torch.cuda.is_available()
    if cfg.cuda:
        net = Net(cfg.scale_factor).to(cfg.device)
    if len(device_ids)>1:
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        
    # load the edge detector model
    model = bdcn.BDCN().to(cfg.device)
    model.load_state_dict(torch.load('%s' % (cfg.edge_model_path)))
    model.eval()

    cudnn.benchmark = True
    scale = cfg.scale_factor
    
    if cfg.load_pretrain_modelx2:
        if os.path.isfile(cfg.modelx2_path):
            print("==> load the pre-trained modelx2_dict")
            model_dict = net.state_dict()
            pretrained_dict = torch.load(cfg.modelx2_path, map_location={'cuda:0': cfg.device})
            pretrained_dict['state_dict'].pop('module.upscale.0.weight')
            pretrained_dict['state_dict'].pop('module.upscale.0.bias')
            pretrained_dict['state_dict'].pop('module.upscale.2.weight')
            pretrained_dict['state_dict'].pop('module.upscale.2.bias')
            pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
        else:
            print("=> no modelx2 found at '{}'".format(cfg.load_model))

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    criterion_L1 = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    loss_epoch = []
    loss_list = []
    file = open(cfg.checkpoints_dir +'/loss.txt','w')
    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):

        for idx_iter, (HR_left, HR_right, LR_left, LR_right, HR_left_e, HR_right_e, LR_left_e, LR_right_e) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            HR_left, HR_right, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device),\
                                                    Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            HR_left_e, HR_right_e, LR_left_e, LR_right_e = Variable(HR_left_e).to(cfg.device), Variable(HR_right_e).to(cfg.device),\
                                                    Variable(LR_left_e).to(cfg.device), Variable(LR_right_e).to(cfg.device)
            with torch.no_grad():
                edge_LR_left = model(LR_left_e)
                edge_LR_right = model(LR_right_e)
                out = [torch.sigmoid(edge_LR_left[-1]).cpu().data.numpy()[0, 0, :, :]]
                edge_LR_left = [torch.sigmoid(x) for x in edge_LR_left]
                edge_LR_right = [torch.sigmoid(x) for x in edge_LR_right]

            SR_left, SR_right = net(LR_left, LR_right, edge_LR_left, edge_LR_right)

            ''' SR Loss '''
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)

            ''' Total Loss '''
            loss = loss_SR
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.data.cpu())

        scheduler.step()

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))

            print('Epoch--%4d, loss--%f'%(idx_epoch + 1, float(np.array(loss_epoch).mean())))
            if idx_epoch == 0 or idx_epoch >= 78:
                torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()},
                           cfg.checkpoints_dir + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_epoch' + str(idx_epoch + 1) + '.pth.tar')
            file.write('Epoch--%4d, loss--%f'%(idx_epoch + 1, float(np.array(loss_epoch).mean()))+'\n')
            file.flush()
            loss_epoch = []
    file.close()


def main(cfg):
    train_set = TrainSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

