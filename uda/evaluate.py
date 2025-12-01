import os
import torch
import numpy as np
import ever as er
import math
from tqdm import tqdm
from data.Hi_UCD.hiucd import HiUCDLoader
from data.second.second import SecondLoader
from data.wusu.wusu import WUSULoader
from PIL import Image
import matplotlib.pyplot as plt


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    c = a[k]
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class=7):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def score_summary(hist):
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    a = hist.sum(1)
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    kappa_n1 = cal_kappa(hist_n0)
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3 * IoU_mean + 0.7 * Sek

    return dict(kappa=kappa_n0,
                mIoU=IoU_mean,
                Sek=Sek,
                Score=Score,
                IoU_1=IoU_fg)


def evaluate_second_stat(config, model_dir, logger, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device =  torch.device('cpu')
    torch.cuda.empty_cache()
    model.eval()
    test_dataloader = SecondLoader(config.data.test.params)
    number_class = 6
    num_changeclass = number_class * number_class
    mulcls_metric_op = er.metric.PixelMetric(num_changeclass,
                                             model_dir,
                                             logger=logger)

    for img12, gt in tqdm(test_dataloader):
        change_true = gt[2].cpu().numpy()
        change_true_op = change_true.ravel()
        valid_inds = np.where(change_true_op == 1)

        refined_pred = model(img12.to(device), postprocess=False)
        refined_pred = refined_pred[:, number_class * 2: number_class * 2 + number_class ** 2, :, :].argmax(dim=1).to(
            dtype=torch.int32)
        refined_pred = refined_pred.cpu().numpy().ravel()[valid_inds]
        target = gt[0] * number_class + gt[1]
        target = target.cpu().numpy()
        target = target.ravel()[valid_inds]
        mulcls_metric_op.forward(target, refined_pred)
    mulcls_metric_op.summary_all()


def evaluate_second_Sek(config, model_dir, logger, model):
    os.makedirs("{}/result/change".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/semantic1".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/semantic2".format(model_dir), exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model.eval()
    test_dataloader = SecondLoader(config.data.test.params)

    number_class = 6
    num_changeclass = number_class * number_class + 1
    hist = np.zeros((num_changeclass, num_changeclass))

    change_metric_op = er.metric.PixelMetric(2,
                                             model_dir,
                                             logger=logger)

    semantic_metric_op = er.metric.PixelMetric(number_class,
                                               model_dir,
                                               logger=logger)

    for img12, gt in tqdm(test_dataloader):
        semantic1_true = gt[0].cpu().numpy()
        semantic2_true = gt[1].cpu().numpy()
        change_true = gt[2].cpu().numpy()
        filename = gt[3]

        change_true = change_true.ravel()
        valid_inds = np.where(change_true != 0)[0]
        semantic1_true = semantic1_true.ravel()[valid_inds]
        semantic2_true = semantic2_true.ravel()[valid_inds]

        refined_pred, pred = model(img12.to(device))
        refined_pred = refined_pred.cpu().numpy()
        target = torch.where(gt[0] >= 0, gt[0] * number_class + gt[1] + 1, torch.zeros_like(gt[0]))
        hist += get_hist(refined_pred, target.cpu().numpy(), num_changeclass)

        change_pred = pred[2:3, :, :].cpu().numpy()
        semantic1_pred = pred[0:1, :, :].cpu().numpy().astype(int)
        semantic2_pred = pred[1:2, :, :].cpu().numpy().astype(int)

        change_pred_op = change_pred.ravel()
        semantic1_pred_op = semantic1_pred.ravel()[valid_inds]
        semantic2_pred_op = semantic2_pred.ravel()[valid_inds]

        change_metric_op.forward(change_true, change_pred_op)
        semantic_metric_op.forward(semantic1_true, semantic1_pred_op)
        semantic_metric_op.forward(semantic2_true, semantic2_pred_op)

        # image
        image_change = np.where(change_pred > 0, 255, 0).astype(np.uint8)
        image_change = Image.fromarray(image_change[0])
        image_change.save("{}/result/change/{}".format(model_dir, filename[0]))

        semantic1_pred_image = np.where(change_pred == 0, 0, semantic1_pred + 1)
        image_semantic1 = Image.fromarray(semantic1_pred_image[0].astype(np.uint8))
        image_semantic1.putpalette([255, 255, 255,
                                    0, 128, 0,
                                    128, 128, 128,
                                    0, 255, 0,
                                    0, 0, 255,
                                    128, 0, 0,
                                    255, 0, 0])
        image_semantic1.convert("RGB")
        image_semantic1.save("{}/result/semantic1/{}".format(model_dir, filename[0]))

        semantic2_pred_image = np.where(change_pred == 0, 0, semantic2_pred + 1)
        image_semantic2 = Image.fromarray(semantic2_pred_image[0].astype(np.uint8))
        image_semantic2.putpalette([255, 255, 255,
                                    0, 128, 0,
                                    128, 128, 128,
                                    0, 255, 0,
                                    0, 0, 255,
                                    128, 0, 0,
                                    255, 0, 0])
        image_semantic2.convert("RGB")
        image_semantic2.save("{}/result/semantic2/{}".format(model_dir, filename[0]))
        #

    change_metric_op.summary_all()
    semantic_metric_op.summary_all()

    eval_dict = score_summary(hist)
    for k, v in eval_dict.items():
        logger.info(f'{k} = %.5f' % v)

    torch.cuda.empty_cache()
    return 



def evaluate_hiucd_stat(config, model_dir, logger, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model.eval()
    test_dataloader = HiUCDLoader(config.data.test.params)
    number_class = 9
    num_changeclass = number_class * number_class
    mulcls_metric_op = er.metric.PixelMetric(num_changeclass,
                                             model_dir,
                                             logger=logger)

    for img12, gt in tqdm(test_dataloader):
        change_true = gt[2].cpu().numpy()
        change_true_op = change_true.ravel()
        valid_inds = np.where(change_true_op != -1)

        refined_pred = model(img12.to(device), postprocess=False)
        refined_pred = refined_pred[:, number_class * 2: number_class * 2 + number_class ** 2, :, :].argmax(dim=1).to(
            dtype=torch.int32)
        refined_pred = refined_pred.cpu().numpy().ravel()[valid_inds]
        target = gt[0] * number_class + gt[1]
        target = target.cpu().numpy()
        target = target.ravel()[valid_inds]
        mulcls_metric_op.forward(target, refined_pred)

    mulcls_metric_op.summary_all()


def evaluate_hiucd_Sek(config, model_dir, logger, model):
    os.makedirs("{}/result/change".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/semantic1".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/semantic2".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/semantic1_SCD".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/semantic2_SCD".format(model_dir), exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model.eval()
    test_dataloader = HiUCDLoader(config.data.test.params)

    number_class = 9
    num_changeclass = number_class * number_class + 1
    hist = np.zeros((num_changeclass, num_changeclass))

    change_metric_op = er.metric.PixelMetric(2,
                                             model_dir,
                                             logger=logger)

    semantic_metric_op = er.metric.PixelMetric(number_class,
                                               model_dir,
                                               logger=logger)

    semantic_metric_op_change = er.metric.PixelMetric(number_class,
                                                      model_dir,
                                                      logger=logger)

    semantic_metric_op_unchange = er.metric.PixelMetric(number_class,
                                                        model_dir,
                                                        logger=logger)

    for img12, gt in tqdm(test_dataloader):
        semantic1_true = gt[0].cpu().numpy()
        semantic2_true = gt[1].cpu().numpy()
        change_true = gt[2].cpu().numpy()
        filename = gt[3]

        change_true_op = change_true.ravel()
        valid_inds = np.where(change_true_op != -1)
        change_inds = np.where(change_true_op == 1)
        unchange_inds = np.where(change_true_op == 0)
        semantic1_true_change = semantic1_true.ravel()[change_inds]
        semantic2_true_change = semantic2_true.ravel()[change_inds]
        semantic1_true_unchange = semantic1_true.ravel()[unchange_inds]
        semantic2_true_unchange = semantic2_true.ravel()[unchange_inds]
        semantic1_true = semantic1_true.ravel()[valid_inds]
        semantic2_true = semantic2_true.ravel()[valid_inds]
        change_true_op = change_true_op.ravel()[[valid_inds]]

        refined_pred, pred = model(img12.to(device))
        refined_pred = refined_pred.cpu().numpy()
        target = torch.where(gt[2] >= 1, gt[0] * number_class + gt[1] + 1, torch.zeros_like(gt[0])).cpu().numpy()
        target_op = target.ravel()[valid_inds]
        refined_pred_op = refined_pred.ravel()[valid_inds]
        hist += get_hist(refined_pred_op, target_op, num_changeclass)

        change_pred = pred[2:3, :, :].cpu().numpy()
        semantic1_pred = pred[0:1, :, :].cpu().numpy().astype(int)
        semantic2_pred = pred[1:2, :, :].cpu().numpy().astype(int)

        change_pred_op = change_pred.ravel()[valid_inds]
        semantic1_pred_op = semantic1_pred.ravel()[valid_inds]
        semantic2_pred_op = semantic2_pred.ravel()[valid_inds]
        semantic1_pred_op_change = semantic1_pred.ravel()[change_inds]
        semantic2_pred_op_change = semantic2_pred.ravel()[change_inds]
        semantic1_pred_op_unchange = semantic1_pred.ravel()[unchange_inds]
        semantic2_pred_op_unchange = semantic2_pred.ravel()[unchange_inds]

        change_metric_op.forward(change_true_op, change_pred_op)
        semantic_metric_op.forward(semantic1_true, semantic1_pred_op)
        semantic_metric_op.forward(semantic2_true, semantic2_pred_op)
        semantic_metric_op_change.forward(semantic1_true_change, semantic1_pred_op_change)
        semantic_metric_op_change.forward(semantic2_true_change, semantic2_pred_op_change)
        semantic_metric_op_unchange.forward(semantic1_true_unchange, semantic1_pred_op_unchange)
        semantic_metric_op_unchange.forward(semantic2_true_unchange, semantic2_pred_op_unchange)

        # image
        image_change = np.where(change_true == -1, -1, change_pred).astype(np.uint8) + 1
        image_change = Image.fromarray(image_change[0])
        image_change.putpalette([255, 255, 255,
                                 0, 0, 0,
                                 255, 0, 0])
        image_change.convert("RGB")
        image_change.save("{}/result/change/{}".format(model_dir, filename[0]))

        semantic1_pred_image = np.where(change_true == -1, 0, semantic1_pred + 1)
        image_semantic1 = Image.fromarray(semantic1_pred_image[0].astype(np.uint8))
        image_semantic1.putpalette([255, 255, 255,
                                    0, 150, 255,
                                    200, 255, 120,
                                    255, 0, 0,
                                    255, 0, 255,
                                    255, 255, 0,
                                    255, 180, 180,
                                    0, 255, 255,
                                    180, 120, 255,
                                    0, 255, 0, ])
        image_semantic1.convert("RGB")
        image_semantic1.save("{}/result/semantic1/{}".format(model_dir, filename[0]))

        semantic2_pred_image = np.where(change_true == -1, 0, semantic2_pred + 1)
        image_semantic2 = Image.fromarray(semantic2_pred_image[0].astype(np.uint8))
        image_semantic2.putpalette([255, 255, 255,
                                    0, 150, 255,
                                    200, 255, 120,
                                    255, 0, 0,
                                    255, 0, 255,
                                    255, 255, 0,
                                    255, 180, 180,
                                    0, 255, 255,
                                    180, 120, 255,
                                    0, 255, 0, ])
        image_semantic2.convert("RGB")
        image_semantic2.save("{}/result/semantic2/{}".format(model_dir, filename[0]))

        semantic1_pred_image = np.where((change_pred == 0), 0, semantic1_pred + 1)
        image_semantic1 = Image.fromarray(semantic1_pred_image[0].astype(np.uint8))
        image_semantic1.putpalette([255, 255, 255,
                                    0, 150, 255,
                                    200, 255, 120,
                                    255, 0, 0,
                                    255, 0, 255,
                                    255, 255, 0,
                                    255, 180, 180,
                                    0, 255, 255,
                                    180, 120, 255,
                                    0, 255, 0, ])
        image_semantic1.convert("RGB")
        image_semantic1.save("{}/result/semantic1_SCD/{}".format(model_dir, filename[0]))

        semantic2_pred_image = np.where((change_pred == 0), 0, semantic2_pred + 1)
        image_semantic2 = Image.fromarray(semantic2_pred_image[0].astype(np.uint8))
        image_semantic2.putpalette([255, 255, 255,
                                    0, 150, 255,
                                    200, 255, 120,
                                    255, 0, 0,
                                    255, 0, 255,
                                    255, 255, 0,
                                    255, 180, 180,
                                    0, 255, 255,
                                    180, 120, 255,
                                    0, 255, 0, ])
        image_semantic2.convert("RGB")
        image_semantic2.save("{}/result/semantic2_SCD/{}".format(model_dir, filename[0]))
        #

    try:
        changestat = model.module.change_stat
    except:
        changestat = model.change_stat

    np.save(os.path.join(model_dir, "stat.npy"), changestat)

    semantic_metric_op_change.summary_all()
    semantic_metric_op_unchange.summary_all()
    change_metric_op.summary_all()
    semantic_metric_op.summary_all()

    eval_dict = score_summary(hist)
    for k, v in eval_dict.items():
        logger.info(f'{k} = %.5f' % v)

    torch.cuda.empty_cache()
    return 


def evaluate_levircd(config, model_dir, logger, model):
    os.makedirs("{}/result/change".format(model_dir), exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model.eval()
    test_dataloader = LEVIRCDLoader(config.data.test.params)

    det_metric_op = er.metric.PixelMetric(2,
                                          model_dir,
                                          logger=logger)

    for img, ret_gt in tqdm(test_dataloader):
        img = img.to(device)

        _, y1y2change = model(img)

        change_pred = y1y2change[-1:, :, :].cpu().numpy()

        image_change = np.where(change_pred > 0, 255, 0).astype(np.uint8)
        image_change = Image.fromarray(image_change[0])
        name = ret_gt["image_filename"][0]
        image_change.save("{}/result/change/{}".format(model_dir, name))

        gt_change = ret_gt['change']
        gt_change = gt_change.numpy()
        y_true = gt_change.ravel()
        y_pred = change_pred.ravel()

        y_true = np.where(y_true > 0, np.ones_like(y_true), np.zeros_like(y_true))

        det_metric_op.forward(y_true, y_pred)

    split = [s.replace('./LEVIR-CD/', '') for s in test_dataloader.config.root_dir]
    split_str = ','.join(split)
    logger.info(f'det -[LEVIRCD {split_str}]')
    det_metric_op.summary_all()

    torch.cuda.empty_cache()


def evaluate_wusu_stat(config, model_dir, logger, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model.eval()
    test_dataloader = WUSULoader(config.data.test.params)
    number_class = 12
    num_changeclass = number_class * number_class
    mulcls_metric_op = er.metric.PixelMetric(num_changeclass,
                                             model_dir,
                                             logger=logger)

    for img12, gt in tqdm(test_dataloader):
        change_true = gt[2].cpu().numpy()
        change_true_op = change_true.ravel()
        valid_inds = np.where(change_true_op != -1)

        refined_pred = model(img12.to(device), postprocess=False)
        refined_pred = refined_pred[:, number_class * 2: number_class * 2 + number_class ** 2, :, :].argmax(dim=1).to(
            dtype=torch.int32)
        refined_pred = refined_pred.cpu().numpy().ravel()[valid_inds]
        target = gt[0] * number_class + gt[1]
        target = target.cpu().numpy()
        target = target.ravel()[valid_inds]
        mulcls_metric_op.forward(target, refined_pred)

    mulcls_metric_op.summary_all()


def evaluate_wusu_Sek(config, model_dir, logger, model):
    os.makedirs("{}/result/HS/change".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/HS/class".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/JA/change".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/JA/class".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/HS/class_SCD".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/JA/class_SCD".format(model_dir), exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model.eval()
    test_dataloader = WUSULoader(config.data.test.params)

    number_class = 12
    num_changeclass = number_class * number_class + 1
    hist = np.zeros((num_changeclass, num_changeclass))

    change_metric_op = er.metric.PixelMetric(2,
                                             model_dir,
                                             logger=logger)

    semantic_metric_op = er.metric.PixelMetric(number_class,
                                               model_dir,
                                               logger=logger)

    semantic_metric_op_change = er.metric.PixelMetric(number_class,
                                                      model_dir,
                                                      logger=logger)

    semantic_metric_op_unchange = er.metric.PixelMetric(number_class,
                                                        model_dir,
                                                        logger=logger)

    mulcls_metric_op = er.metric.PixelMetric(num_changeclass,
                                             model_dir,
                                             logger=logger)

    for img12, gt in tqdm(test_dataloader):
        semantic1_true = gt[0].cpu().numpy()
        semantic2_true = gt[1].cpu().numpy()
        change_true = gt[2].cpu().numpy()
        filename1 = gt[3][0]
        filename2 = gt[4][0]
        loc = filename1[:2]
        time1 = filename1[2:4]
        time2 = filename2[2:4]
        num = filename1[5:][:-4]

        change_true_op = change_true.ravel()
        valid_inds = np.where(change_true_op != -1)
        change_inds = np.where(change_true_op == 1)
        unchange_inds = np.where(change_true_op == 0)
        semantic1_true_change = semantic1_true.ravel()[change_inds]
        semantic2_true_change = semantic2_true.ravel()[change_inds]
        semantic1_true_unchange = semantic1_true.ravel()[unchange_inds]
        semantic2_true_unchange = semantic2_true.ravel()[unchange_inds]
        semantic1_true = semantic1_true.ravel()[valid_inds]
        semantic2_true = semantic2_true.ravel()[valid_inds]
        change_true_op = change_true_op.ravel()[[valid_inds]]

        refined_pred, pred = model(img12.to(device))
        refined_pred = refined_pred.cpu().numpy()
        target = torch.where(gt[2] >= 1, gt[0] * number_class + gt[1] + 1, torch.zeros_like(gt[0])).cpu().numpy()
        target_op = target.ravel()[valid_inds]
        refined_pred_op = refined_pred.ravel()[valid_inds]
        hist += get_hist(refined_pred_op, target_op, num_changeclass)

        change_pred = pred[2:3, :, :].cpu().numpy()
        semantic1_pred = pred[0:1, :, :].cpu().numpy().astype(int)
        semantic2_pred = pred[1:2, :, :].cpu().numpy().astype(int)

        change_pred_op = change_pred.ravel()[valid_inds]
        semantic1_pred_op = semantic1_pred.ravel()[valid_inds]
        semantic2_pred_op = semantic2_pred.ravel()[valid_inds]
        semantic1_pred_op_change = semantic1_pred.ravel()[change_inds]
        semantic2_pred_op_change = semantic2_pred.ravel()[change_inds]
        semantic1_pred_op_unchange = semantic1_pred.ravel()[unchange_inds]
        semantic2_pred_op_unchange = semantic2_pred.ravel()[unchange_inds]

        change_metric_op.forward(change_true_op, change_pred_op)
        semantic_metric_op.forward(semantic1_true, semantic1_pred_op)
        semantic_metric_op.forward(semantic2_true, semantic2_pred_op)
        semantic_metric_op_change.forward(semantic1_true_change, semantic1_pred_op_change)
        semantic_metric_op_change.forward(semantic2_true_change, semantic2_pred_op_change)
        semantic_metric_op_unchange.forward(semantic1_true_unchange, semantic1_pred_op_unchange)
        semantic_metric_op_unchange.forward(semantic2_true_unchange, semantic2_pred_op_unchange)

        # image
        image_change = np.where(change_true == -1, -1, change_pred).astype(np.uint8) + 1
        image_change = Image.fromarray(image_change[0])
        image_change.putpalette([255, 255, 255,
                                 128, 128, 128,
                                 0, 0, 255])
        image_change.convert("RGB")
        image_change.save("{}/result/{}/change/{}{}{}_{}.png".format(model_dir, loc, loc, time1, time2, num))

        semantic1_pred_image = np.where(change_true == -1, 0, semantic1_pred + 1)
        image_semantic1 = Image.fromarray(semantic1_pred_image[0].astype(np.uint8))
        image_semantic1.putpalette([255, 255, 255,
                                    0, 0, 0,
                                    255, 211, 127,
                                    255, 0, 0,
                                    255, 235, 175,
                                    0, 0, 0,
                                    38, 116, 0,
                                    170, 255, 0,
                                    0, 197, 255,
                                    0, 92, 230,
                                    0, 255, 197,
                                    197, 0, 255,
                                    178, 178, 178,
                                    ])
        image_semantic1.convert("RGB")
        image_semantic1.save("{}/result/{}/class/{}{}_{}.png".format(model_dir, loc, loc, time1, num))

        semantic2_pred_image = np.where(change_true == -1, 0, semantic2_pred + 1)
        image_semantic2 = Image.fromarray(semantic2_pred_image[0].astype(np.uint8))
        image_semantic2.putpalette([255, 255, 255,
                                    0, 0, 0,
                                    255, 211, 127,
                                    255, 0, 0,
                                    255, 235, 175,
                                    0, 0, 0,
                                    38, 116, 0,
                                    170, 255, 0,
                                    0, 197, 255,
                                    0, 92, 230,
                                    0, 255, 197,
                                    197, 0, 255,
                                    178, 178, 178,
                                    ])
        image_semantic2.convert("RGB")
        image_semantic2.save("{}/result/{}/class/{}{}_{}.png".format(model_dir, loc, loc, time2, num))

        semantic1_pred_image = np.where(change_pred == 0, 0, semantic1_pred + 1)
        semantic1_pred_image = np.where(change_true == -1, 0, semantic1_pred_image)
        semantic1_pred_image = np.where((change_true>=0)&(change_pred==0), 13, semantic1_pred_image)
        image_semantic1 = Image.fromarray(semantic1_pred_image[0].astype(np.uint8))
        image_semantic1.putpalette([255, 255, 255,
                                    0, 0, 0,
                                    255, 211, 127,
                                    255, 0, 0,
                                    255, 235, 175,
                                    0, 0, 0,
                                    38, 116, 0,
                                    170, 255, 0,
                                    0, 197, 255,
                                    0, 92, 230,
                                    0, 255, 197,
                                    197, 0, 255,
                                    178, 178, 178,
                                    128,128,128
                                    ])
        image_semantic1.convert("RGB")
        image_semantic1.save("{}/result/{}/class_SCD/{}{}_{}.png".format(model_dir, loc, loc, time1, num))

        semantic2_pred_image = np.where(change_pred == 0, 0, semantic2_pred + 1)
        semantic2_pred_image = np.where(change_true == -1, 0, semantic2_pred_image)
        semantic2_pred_image = np.where((change_true>=0)&(change_pred==0), 13, semantic2_pred_image)
        image_semantic2 = Image.fromarray(semantic2_pred_image[0].astype(np.uint8))
        image_semantic2.putpalette([255, 255, 255,
                                    0, 0, 0,
                                    255, 211, 127,
                                    255, 0, 0,
                                    255, 235, 175,
                                    0, 0, 0,
                                    38, 116, 0,
                                    170, 255, 0,
                                    0, 197, 255,
                                    0, 92, 230,
                                    0, 255, 197,
                                    197, 0, 255,
                                    178, 178, 178,
                                    128,128,128
                                    ])
        image_semantic2.convert("RGB")
        image_semantic2.save("{}/result/{}/class_SCD/{}{}_{}.png".format(model_dir, loc, loc, time2, num))
        #
    try:
        changestat = model.module.change_stat
    except:
        changestat = model.change_stat

    np.save(os.path.join(model_dir, "stat.npy"), changestat)

    semantic_metric_op_change.summary_all()
    semantic_metric_op_unchange.summary_all()
    change_metric_op.summary_all()
    semantic_metric_op.summary_all()

    eval_dict = score_summary(hist)
    for k, v in eval_dict.items():
        logger.info(f'{k} = %.5f' % v)

    torch.cuda.empty_cache()

def evaluate_hrscd_Sek(config, model_dir, logger, model):
    os.makedirs("{}/result/change".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/semantic1".format(model_dir), exist_ok=True)
    os.makedirs("{}/result/semantic2".format(model_dir), exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    model.eval()
    test_dataloader = HRSCDLoader(config.data.test.params)

    number_class = 5
    num_changeclass = number_class * number_class + 1
    hist = np.zeros((num_changeclass, num_changeclass))

    change_metric_op = er.metric.PixelMetric(2,
                                             model_dir,
                                             logger=logger)

    semantic_metric_op = er.metric.PixelMetric(number_class,
                                               model_dir,
                                               logger=logger)

    semantic_metric_op_change = er.metric.PixelMetric(number_class,
                                                      model_dir,
                                                      logger=logger)

    semantic_metric_op_unchange = er.metric.PixelMetric(number_class,
                                                        model_dir,
                                                        logger=logger)

    for img12, gt in tqdm(test_dataloader):
        semantic1_true = gt[0].cpu().numpy()
        semantic2_true = gt[1].cpu().numpy()
        change_true = gt[2].cpu().numpy()
        filename = gt[3]

        change_true_op = change_true.ravel()
        valid_inds = np.where(change_true_op != -1)
        change_inds = np.where(change_true_op == 1)
        unchange_inds = np.where(change_true_op == 0)
        semantic1_true_change = semantic1_true.ravel()[change_inds]
        semantic2_true_change = semantic2_true.ravel()[change_inds]
        semantic1_true_unchange = semantic1_true.ravel()[unchange_inds]
        semantic2_true_unchange = semantic2_true.ravel()[unchange_inds]
        semantic1_true = semantic1_true.ravel()[valid_inds]
        semantic2_true = semantic2_true.ravel()[valid_inds]
        change_true_op = change_true_op.ravel()[[valid_inds]]

        refined_pred, pred = model(img12.to(device))
        refined_pred = refined_pred.cpu().numpy()
        target = torch.where(gt[2] >= 1, gt[0] * number_class + gt[1] + 1, torch.zeros_like(gt[0])).cpu().numpy()
        target_op = target.ravel()[valid_inds]
        refined_pred_op = refined_pred.ravel()[valid_inds]
        hist += get_hist(refined_pred_op, target_op, num_changeclass)

        change_pred = pred[-1:, :, :].cpu().numpy()
        semantic1_pred = pred[0:1, :, :].cpu().numpy().astype(int)
        semantic2_pred = pred[1:2, :, :].cpu().numpy().astype(int)

        change_pred_op = change_pred.ravel()[valid_inds]
        semantic1_pred_op = semantic1_pred.ravel()[valid_inds]
        semantic2_pred_op = semantic2_pred.ravel()[valid_inds]
        semantic1_pred_op_change = semantic1_pred.ravel()[change_inds]
        semantic2_pred_op_change = semantic2_pred.ravel()[change_inds]
        semantic1_pred_op_unchange = semantic1_pred.ravel()[unchange_inds]
        semantic2_pred_op_unchange = semantic2_pred.ravel()[unchange_inds]

        change_metric_op.forward(change_true_op, change_pred_op)
        semantic_metric_op.forward(semantic1_true, semantic1_pred_op)
        semantic_metric_op.forward(semantic2_true, semantic2_pred_op)
        semantic_metric_op_change.forward(semantic1_true_change, semantic1_pred_op_change)
        semantic_metric_op_change.forward(semantic2_true_change, semantic2_pred_op_change)
        semantic_metric_op_unchange.forward(semantic1_true_unchange, semantic1_pred_op_unchange)
        semantic_metric_op_unchange.forward(semantic2_true_unchange, semantic2_pred_op_unchange)

        # image
        image_change = np.where(change_true == -1, 0, change_pred).astype(np.uint8)
        image_change = Image.fromarray(image_change[0])
        image_change.putpalette([0,0,0,
                          235,86,154])
        image_change.convert("RGB")
        image_change.save("{}/result/change/{}".format(model_dir, filename[0].replace("tif","png")))

        semantic1_pred_image = np.where(change_true == -1, 0, semantic1_pred + 1)
        image_semantic1 = Image.fromarray(semantic1_pred_image[0].astype(np.uint8))
        image_semantic1.putpalette([0,0,0,
                          20,20,230,
                          200,100,50,
                          0,255,0,
                          172,221,165,
                          44,131,18])
        image_semantic1.convert("RGB")
        image_semantic1.save("{}/result/semantic1/{}".format(model_dir, filename[0].replace("tif","png")))

        semantic2_pred_image = np.where(change_true == -1, 0, semantic2_pred + 1)
        image_semantic2 = Image.fromarray(semantic2_pred_image[0].astype(np.uint8))
        image_semantic2.putpalette([0,0,0,
                          20,20,230,
                          200,100,50,
                          0,255,0,
                          172,221,165,
                          44,131,18])
        image_semantic2.convert("RGB")
        image_semantic2.save("{}/result/semantic2/{}".format(model_dir, filename[0].replace("tif","png")))
        #

    try:
        changestat = model.module.change_stat
    except:
        changestat = model.change_stat

    np.save(os.path.join(model_dir, "stat.npy"), changestat)

    semantic_metric_op_change.summary_all()
    semantic_metric_op_unchange.summary_all()
    change_metric_op.summary_all()
    semantic_metric_op.summary_all()

    eval_dict = score_summary(hist)
    for k, v in eval_dict.items():
        logger.info(f'{k} = %.5f' % v)

    logger.info(changestat)
    x = np.arange(81)
    plt.cla()
    plt.plot(x, changestat)
    plt.savefig("{}/result/change_type.png".format(model_dir))

    torch.cuda.empty_cache()