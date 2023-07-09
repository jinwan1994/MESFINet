from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
from model import *
import cv2
import bdcn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='/home/wanjin/dataset/test/')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='MESFINet_2xSR')
    parser.add_argument('--dataset_list', type=str, default='KITTI2012+KITTI2015+Middlebury')
    parser.add_argument('--sr_model_path', type=str, default='./log/MESFINet_2xSR_final.pth.tar')
    parser.add_argument('--edge_model_path', type=str, default='./pretrained_moel/bdcn_pretrained_on_nyudv2_rgb.pth',
        help='the model to get the edge prior')
    return parser.parse_args()


def test(cfg):
    net = Net(cfg.scale_factor).to(cfg.device)
    net = torch.nn.DataParallel(net)
    net_model = torch.load(cfg.sr_model_path)
    net.load_state_dict(net_model['state_dict'])
    model = bdcn.BDCN().to(cfg.device)
    model.load_state_dict(torch.load('%s' % (cfg.edge_model_path)))
    net.eval()
    model.eval()
        
    file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor))
    for idx in range(len(file_list)):
        LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr0.png')
        LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr1.png')
        with torch.no_grad():
            Edge_LR_left, Edge_LR_right = toEdge_Tensor(LR_left), toEdge_Tensor(LR_right)
            Edge_LR_left, Edge_LR_right = Edge_LR_left.unsqueeze(0), Edge_LR_right.unsqueeze(0)
            Edge_LR_left, Edge_LR_right = Variable(Edge_LR_left).to(cfg.device), Variable(Edge_LR_right).to(cfg.device)
        
            LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
            LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
            LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            scene_name = file_list[idx]
        print('Running Scene ' + scene_name + ' of ' + cfg.dataset + ' Dataset......')
        with torch.no_grad():

            edge_LR_left = model(Edge_LR_left)
            edge_LR_right = model(Edge_LR_right)
            edge_LR_left = [torch.sigmoid(x) for x in edge_LR_left]
            edge_LR_right = [torch.sigmoid(x) for x in edge_LR_right]
            
            SR_left, SR_right = net(LR_left, LR_right, edge_LR_left, edge_LR_right)
            SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
        save_path = cfg.save_dir + '/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L.png')
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R.png')

def toEdge_Tensor(img):
    img = np.array(img, np.float32)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float()

if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = cfg.dataset_list.split('+')
    for i in range(len(dataset_list)):
        cfg.dataset = dataset_list[i]
        test(cfg)
    print('Finished!')
