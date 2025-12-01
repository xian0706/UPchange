import argparse
import torch
import torch.optim as optim
import os.path as osp
import math
from module.UPchange.UPchange import UPchange
from data.second.second_single import SecondLoader_single
from data.second.second import SecondLoader
from ever.core.iterator import Iterator
from ever.core import to
from uda.tools import *
from uda.evaluate import evaluate_second_Sek, evaluate_second_stat
from uda.Second_ST import SecondLoader_ST
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.nn.utils import clip_grad

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

er.registry.register_all()

parser = argparse.ArgumentParser(description='Run ChangeUDA methods.')

parser.add_argument('--config_path', type=str,default="module.UPchange.UPchange_second",
                    help='config path')
parser.add_argument('--model_dir',default=None,type=str)
parser.add_argument('--test',default=False,type=bool)
parser.add_argument('--ratio', type=float,default=0.5)

args = parser.parse_args()
cfg = import_config(args.config_path, args.model_dir)


def remove_module_prefix(model_state_dict):
    ret = {}
    safe_flag = False
    for k, v in model_state_dict.items():
        if 'module.' not in k:
            safe_flag = True
            break
        if k.find('module.changemask.') == 0:
            k = k.replace('module.changemask.', '', 1)
        ret[k] = v
    if safe_flag:
        return model_state_dict
    else:
        return ret


def main():
    os.makedirs(args.model_dir, exist_ok=True)
    logger = get_console_file_logger(name='CBST', logdir=args.model_dir)
    cudnn.enabled = True

    save_pseudo_label_path = osp.join(args.model_dir, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_stats_path = osp.join(args.model_dir, 'stats')  # in 'save_path'

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)

    model = UPchange(cfg.model.params)
    model.train()
    model.cuda()

    count_model_parameters(model, logger)

    trainloader = SecondLoader_single(cfg.data.train.params)
    trainloader_iter = Iterator(trainloader)
    evalloader = SecondLoader(cfg.data.test.params)
    targetloader = None
    # print(len(trainloader))
    # print(len(evalloader))

    epochs = cfg.train.num_iters / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = optim.SGD(model.changemask.parameters(),
                          lr=cfg.learning_rate.params.base_lr, momentum=cfg.optimizer.params.momentum,
                          weight_decay=cfg.optimizer.params.weight_decay)
    optimizer.zero_grad()

    if args.test:
        # weight = torch.load(osp.join(args.model_dir[:-2],"33", "model-" + str(10000) + '.pth'))
        # weight = remove_module_prefix(weight["model"])
        weight = torch.load(osp.join(args.model_dir, "model-" + str(25000) + '.pth'))
        model.changemask.load_state_dict(weight)
        evaluate_second_Sek(cfg, args.model_dir, logger, model)
        print("eval")

    weight_cls = torch.tensor(np.array([1.0] * cfg.model.params.change_decoder.classifier.out_channels),
                              dtype=torch.float32).cuda()

    # targetloader = SecondLoader_ST(cfg.data.target.params, save_pseudo_label_path)
    # targetloader_iter = Iterator(targetloader)

    for i_iter in tqdm(range(0, cfg.train.num_iters)):
        # i_iter = 19999
        if i_iter < cfg.warmup_step:
            optimizer.zero_grad()
            G_lr = adjust_learning_rate(optimizer, i_iter, cfg)
            d = trainloader_iter.next()
            d = to.to_device(d, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            x, y = d[0]
            loss_seg = model(img_target=None, x=x, y=y, weight=weight_cls)

            total_loss = sum([e for e in loss_seg.values()])
            total_loss.backward()
            loss_seg = {k: v for k, v in loss_seg.items() if k.endswith('loss')}
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.changemask.parameters()), max_norm=35,
                                      norm_type=2)
            optimizer.step()

            if i_iter % 50 == 0:
                logger.info('exp = {}'.format(args.model_dir))
                text = 'iter = %d,  lr = %.3f' % (i_iter, G_lr)
                logger.info(text)
                logger.info(''.join(
                    ['{name} = {value}, '.format(name=name, value=value) for name, value in
                     loss_seg.items()]))
                logger.info(weight_cls)

            if (i_iter + 1) % cfg.eval_every == 0 and i_iter != 0:
                print('save model ...')
                ckpt_path = osp.join(args.model_dir, "model-" + str(i_iter + 1) + '.pth')
                torch.save(model.changemask.state_dict(), ckpt_path)
                evaluate_second_Sek(cfg, args.model_dir, logger, model)
                model.train()
                try:
                    changestat = model.module.change_stat
                except:
                    changestat = model.change_stat
                mean = np.sum(np.log10(changestat + 1)) / 36
                a = np.log10(changestat + 1) / mean
                for i in range(36):
                    weight_cls[i] = torch.tensor(np.array(a[i]), dtype=torch.float32)

        else:
            if (i_iter + 1) % cfg.generate_psedo_every == 0 or targetloader is None:
                logger.info('###### Start generate pesudo dataset in round {}! ######'.format(i_iter))
                save_round_eval_path = osp.join(args.model_dir, str(i_iter))
                if not os.path.exists(save_round_eval_path):
                    os.makedirs(save_round_eval_path)
                # evaluation & save confidence vectors
                conf_dict, pred_cls_num, save_prob_path, save_pred_path, image_name_tgt_list = val(model,
                                                                                                   evalloader,
                                                                                                   save_round_eval_path,
                                                                                                   cfg)

                # class-balanced thresholds
                tgt_portion = (10000 + i_iter) / 50000
                cls = (50000 - i_iter) / 50000
                cls_thresh = kc_parameters(conf_dict, pred_cls_num, tgt_portion, i_iter, save_stats_path, cfg, logger,
                                           cls)
                logger.info(cls_thresh)
                label_selection(cls_thresh, image_name_tgt_list, i_iter, save_prob_path, save_pred_path,
                                save_pseudo_label_path, save_round_eval_path, logger)
                targetloader = SecondLoader_ST(cfg.data.target.params, save_pseudo_label_path)
                targetloader_iter = Iterator(targetloader)
                logger.info('###### Start model retraining dataset in round {}! ######'.format(i_iter))

            model.train()
            G_lr = adjust_learning_rate(optimizer, i_iter, cfg)

            d1 = trainloader_iter.next()
            d1 = to.to_device(d1, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            x1, y1 = d1[0]
            loss_source = model(img_target=None, x=x1, y=y1, weight=weight_cls)

            d2 = targetloader_iter.next()
            d2 = to.to_device(d2, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            x2, y2 = d2[0]
            loss_target = model(img_target=None, x=x2, y=y2, source=False)
            total_loss = sum([a + b * 0.5 for a, b in zip(loss_source.values(), loss_target.values())])

            optimizer.zero_grad()
            total_loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.changemask.parameters()), max_norm=35,
                                      norm_type=2)
            optimizer.step()

            if i_iter % 50 == 0:
                logger.info('exp = {}'.format(args.model_dir))
                text = 'iter = %d,  lr = %.3f' % (i_iter, G_lr)
                logger.info(text)
                logger.info('source:'.join(
                    ['{name} = {value}, '.format(name=name, value=value) for name, value in
                     loss_source.items()]))
                logger.info('target:'.join(
                    ['{name} = {value}, '.format(name=name, value=value) for name, value in
                     loss_target.items()]))
                logger.info(weight_cls)

            if (i_iter + 1) % cfg.eval_every == 0 and i_iter != 0:
                print('save model ...')
                ckpt_path = osp.join(args.model_dir, "model-" + str(i_iter + 1) + '.pth')
                torch.save(model.changemask.state_dict(), ckpt_path)
                evaluate_second_Sek(cfg, args.model_dir, logger, model)
                model.train()
                try:
                    changestat = model.module.change_stat
                except:
                    changestat = model.change_stat
                mean = np.sum(np.log10(changestat + 1)) / 36
                a = np.log10(changestat + 1) / mean
                for i in range(36):
                    weight_cls[i] = torch.tensor(np.array(a[i]), dtype=torch.float32)


def val(model, targetloader, save_round_eval_path, cfg):
    """Create the model and start the evaluation process."""
    model.eval()
    ## output folder
    save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
    save_prob_path = osp.join(save_round_eval_path, 'prob')
    save_pred_path = osp.join(save_round_eval_path, 'pred')

    # viz_op = er.viz.VisualizeSegmm(save_pred_vis_path, palette)
    # metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)

    if not os.path.exists(save_prob_path):
        os.makedirs(save_prob_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    # saving output data
    classes = cfg.model.params.semantic_decoder.classifier.out_channels
    conf_dict = {k: [] for k in range(classes ** 2)}
    pred_cls_num = np.zeros(classes ** 2)
    ## evaluation process
    image_name_tgt_list = []
    with torch.no_grad():
        for batch in tqdm(targetloader):
            images, labels = batch
            output = model(images.cuda(), postprocess=False)[:, classes * 2: classes * 2 + classes ** 2, :, :].softmax(
                dim=1)
            output = output[0] if isinstance(output, tuple) else output
            pred_label = output.argmax(dim=1).cpu().numpy()
            output = output.cpu().numpy()
            for fname, pred_i, out_i in zip(labels[3], pred_label, output):
                image_name_tgt_list.append(fname.split('.')[0])
                # save prob
                # viz_op(pred_i, fname)
                np.save('%s/%s' % (save_prob_path, fname.replace('png', 'npy')), out_i)
                imsave('%s/%s' % (save_pred_path, fname), pred_i.astype(np.uint8), check_contrast=False)
                out_i = out_i.transpose(1, 2, 0)
                conf_i = np.amax(out_i, axis=2)
                # save class-wise confidence maps
                for idx_cls in range(classes ** 2):
                    idx_temp = pred_i == idx_cls
                    pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                    if idx_temp.any():
                        conf_cls_temp = conf_i[idx_temp].astype(np.float32)
                        len_cls_temp = conf_cls_temp.size
                        # downsampling by ds_rate
                        conf_cls = conf_cls_temp[0:len_cls_temp:4]
                        conf_dict[idx_cls].extend(conf_cls)
        return conf_dict, pred_cls_num, save_prob_path, save_pred_path, image_name_tgt_list  # return the dictionary containing all the class-wise confidence vectors
        # conf_dict:一个字典，存储每个类别被选中时的概率。pred_cls_num:一个列表，表示每个类别被选中的个数（均指测试集所有数据）


def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, cfg, logger, cls):
    # conf_dict:一个字典，存储每个类别被选中时的概率。pred_cls_num:一个列表，表示每个类别被选中的个数（均指测试集所有数据）
    classes = cfg.model.params.semantic_decoder.classifier.out_channels
    logger.info('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()
    # threshold for each class
    cls_thresh = np.ones(classes ** 2, dtype=np.float32)
    cls_sel_size = np.zeros(classes ** 2, dtype=np.float32)
    cls_size = np.zeros(classes ** 2, dtype=np.float32)
    # if cfg.KC_POLICY == 'cb' and cfg.KC_VALUE == 'conf':
    for idx_cls in np.arange(0, classes ** 2):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] != None:
            conf_dict[idx_cls].sort(reverse=True)  # sort in descending order
            len_cls = len(conf_dict[idx_cls])
            cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
                if conf_dict[idx_cls][len_cls_thresh - 1] <= cls:
                    cls_thresh[idx_cls] = cls
                else:
                    cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh - 1]
            # 排序后选择每个类别所有被选中概率的中间值
            conf_dict[idx_cls] = None
    return cls_thresh


def label_selection(cls_thresh, image_name_tgt_list, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path,
                    save_round_eval_path, logger):
    logger.info('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()
    # viz_op = er.viz.VisualizeSegmm(save_pseudo_label_color_path, palette)
    for sample_name in tqdm(image_name_tgt_list):
        probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
        pred_prob = np.load(probmap_path)
        weighted_prob = pred_prob / cls_thresh[:, None, None]
        weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=0), dtype=np.uint8)
        weighted_conf = np.amax(weighted_prob, axis=0)
        pred_label_trainIDs = weighted_pred_trainIDs.copy()
        pred1 = pred_label_trainIDs // 6 + 1
        pred2 = pred_label_trainIDs % 6 + 1

        # pred1[pred_label_trainIDs ==0] = 7
        # pred2[pred_label_trainIDs ==0] = 7

        pred1[weighted_conf < 1] = 0
        pred2[weighted_conf < 1] = 0

        # save pseudo-label map with label IDs
        os.makedirs(os.path.join(save_pseudo_label_path, 'label1'), exist_ok=True)
        os.makedirs(os.path.join(save_pseudo_label_path, 'label2'), exist_ok=True)
        imsave(os.path.join(save_pseudo_label_path, 'label1', '%s.png' % sample_name), np.uint8(pred1),
               check_contrast=False)
        imsave(os.path.join(save_pseudo_label_path, 'label2', '%s.png' % sample_name), np.uint8(pred2),
               check_contrast=False)

    # remove probability maps   q
    shutil.rmtree(save_prob_path)

    logger.info('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,
                                                                                                              time.time() - start_pl))


if __name__ == '__main__':
    seed_torch(2333)
    main()
