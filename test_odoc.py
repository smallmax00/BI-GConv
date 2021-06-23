import torch.nn.functional as F
import numpy as np
import torch
import os
import argparse
from lib.ODOC_BMVC import ODOC_seg_edge
from utils.Dataloader_ODOC import ODOC  
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, jaccard_score, balanced_accuracy_score
import cv2
from PIL import Image, ImageFilter
from medpy import metric



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='your_folder/oc_od', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='oc_od/ODOC_BMVC', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=48,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.006,
                    help='maximum epoch number to train')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.1,
                    help='balance factor to control edge and body loss')
parser.add_argument('--alpha', type=float,  default=0,
                    help='balance factor to control consistency loss and body loss')



args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "_{}_bs_beta_{}_base_lr_{}/".format(args.batch_size, args.beta, args.base_lr)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


saved_model_path = os.path.join(snapshot_path, 'best_model.pth')


def calculate_metric_percase(pred, gt):
    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    dice = dice_coef(gt, pred)
    jc = jaccard_score(gt.flatten(), pred.flatten())
    bc = balanced_accuracy_score(gt.flatten(), pred.flatten())
    dice2 = metric.binary.dc(pred, gt)
    return dice, jc, dice2, bc


if __name__ == "__main__":
    model = ODOC_seg_edge()
    model = model.cuda()

    db_test = ODOC(base_dir=train_data_path, split='test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    best_performance = 0.0
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    with torch.no_grad():
        total_metric_cup = 0.0
        total_metric_disc = 0.0

        cup_list = []
        disc_list = []
        name_list = []

        for i_batch, (sampled_batch, sampled_name) in enumerate(testloader):
            volume_batch, label_batch, edge_batch = sampled_batch['img'], sampled_batch['mask'], sampled_batch['con_gau']
            volume_batch, label_batch, edge_batch = volume_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor), edge_batch.type(torch.FloatTensor)
            volume_batch, label_batch, edge_batch = volume_batch.cuda(), label_batch.cuda(), edge_batch.cuda()

            outputs1,  edge_outputs1 = model(volume_batch)
            
            pred_edge = F.upsample(input=edge_outputs1, size=(256, 256), mode='bilinear')
            pred_seg = F.upsample(input=outputs1, size=(256, 256), mode='bilinear')

            # seg
            y_pre = pred_seg.cpu().data.numpy().squeeze()
            y_pre_gt = label_batch.cpu().data.numpy().squeeze()

            y_map_cup = (y_pre[0] > 0.5).astype(np.uint8)
            y_map_disc = (y_pre[1] > 0.5).astype(np.uint8)

            """uncomment below to see a smoothed boundary"""
            # image = Image.fromarray(y_map_cup)
            # filter_image = image.filter(ImageFilter.ModeFilter(size=10))
            # y_map_cup = np.asarray(filter_image)
            # y_map_cup = (y_map_cup > 0).astype(np.uint8)
            # 
            # image = Image.fromarray(y_map_disc)
            # filter_image = image.filter(ImageFilter.ModeFilter(size=10))
            # y_map_disc = np.asarray(filter_image)
            # y_map_disc = (y_map_disc > 0).astype(np.uint8)


            y_map_gt_cup = y_pre_gt[0, ...].astype(np.uint8)
            y_map_gt_disc = y_pre_gt[1, ...].astype(np.uint8)

            single_metric_cup = calculate_metric_percase(y_map_cup, y_map_gt_cup)
            total_metric_cup += np.asarray(single_metric_cup)
            single_metric_disc = calculate_metric_percase(y_map_disc, y_map_gt_disc)
            total_metric_disc += np.asarray(single_metric_disc)
            cup_list.append(single_metric_cup)
            disc_list.append(single_metric_disc)
            name_list.append(sampled_name)


        print('cup_dice_mean:', np.array(cup_list)[:, 0].mean())
        print('disc_dice_mean:', np.array(disc_list)[:, 0].mean())

        print('BC_cup_mean:', np.array(cup_list)[:, 3].mean())
        print('BC_disc_mean:', np.array(disc_list)[:, 3].mean())
            
           

        CI_cup_dice = []
        CI_disc_dice = []
        CI_cup_BC = []
        CI_disc_BC = []
        n_bootstraps = 2000
        rng_seed = 42  # control reproducibility
        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(testloader), len(testloader))
            cup_dice_CI = np.array(cup_list)[indices, 0].mean()
            disc_dice_CI = np.array(disc_list)[indices, 0].mean()
            BC_cup_CI = np.array(cup_list)[indices, 3].mean()
            BC_disc_CI = np.array(disc_list)[indices, 3].mean()
            CI_cup_dice.append(cup_dice_CI)
            CI_disc_dice.append(disc_dice_CI)
            CI_cup_BC.append(BC_cup_CI)
            CI_disc_BC.append(BC_disc_CI)
        # cup_dice_CI
        sorted_scores = np.sort(np.array(CI_cup_dice))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print('95CI_cup_dice, lower:, higher:',confidence_lower, confidence_upper )
        # disc_dice_CI
        sorted_scores = np.sort(np.array(CI_disc_dice))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print('95CI_disc_dice, lower:, higher:', confidence_lower, confidence_upper)
        # BC_cup_CI
        sorted_scores = np.sort(np.array(CI_cup_BC))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print('95CI_cup_BC, lower:, higher:', confidence_lower, confidence_upper)

        # BC_disc_CI
        sorted_scores = np.sort(np.array(CI_disc_BC))
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        print('95CI_disc_BC, lower:, higher:', confidence_lower, confidence_upper)



       
       





