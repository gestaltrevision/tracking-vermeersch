import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter
from mxnet.contrib import amp

from gluoncv.data.transforms import video
from gluoncv.data import UCF101, Kinetics400, SomethingSomethingV2, HMDB51, VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load
from gluoncv.data.sampler import SplitSampler, ShuffleSplitSampler
import os
from bunch import bunchify,Bunch
from lr_schedulers import CosineAnnealingSchedule,CyclicalSchedule
import yaml
from tqdm import tqdm
from pathlib import Path
from parser_helper import parse_args, find_model_params

parser = argparse.ArgumentParser()

parser.add_argument("--default_path", type=str,
                    help="File with defautl config settings for testing")
                    
parser.add_argument("--custom_path", type=str,
                    help="File with current custom config settings for testing")


    
def get_data_loader(opt, batch_size, num_workers, logger, kvstore="None"):
    data_dir = opt.data_dir
    val_data_dir = opt.val_data_dir
    scale_ratios = [float(i) for i in opt.scale_ratios.split(',')]
    input_size = opt.input_size
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]

    def batch_fn(batch, ctx):
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    if opt.data_aug == 'v1':
        # GluonCV style, not keeping aspect ratio, multi-scale crop
        transform_train = video.VideoGroupTrainTransform(size=(input_size, input_size), scale_ratios=scale_ratios,
                                                         more_fix_crop=opt.more_fix_crop, max_distort=opt.max_distort,
                                                         mean=default_mean, std=default_std)
        transform_test = video.VideoGroupValTransform(size=input_size,
                                                      mean=default_mean, std=default_std)
    elif opt.data_aug == 'v2':
        # GluonCV style, keeping aspect ratio, multi-scale crop, same as mmaction style
        transform_train = video.VideoGroupTrainTransformV2(size=(input_size, input_size), short_side=opt.new_height, scale_ratios=scale_ratios,
                                                         mean=default_mean, std=default_std)
        transform_test = video.VideoGroupValTransformV2(crop_size=(input_size, input_size), short_side=opt.new_height,
                                                        mean=default_mean, std=default_std)
    elif opt.data_aug == 'v3':
        # PySlowFast style, keeping aspect ratio, random short side scale jittering
        transform_train = video.VideoGroupTrainTransformV3(crop_size=(input_size, input_size), min_size=opt.new_height, max_size=opt.new_width,
                                                           mean=default_mean, std=default_std)
        transform_test = video.VideoGroupValTransformV2(crop_size=(input_size, input_size), short_side=opt.new_height,
                                                        mean=default_mean, std=default_std)
    elif opt.data_aug == 'v4':
        # mmaction style, keeping aspect ratio, random crop and resize, only for SlowFast family models, similar to 'v3'
        transform_train = video.VideoGroupTrainTransformV4(size=(input_size, input_size),
                                                           mean=default_mean, std=default_std)
        transform_test = video.VideoGroupValTransformV2(crop_size=(input_size, input_size), short_side=opt.new_height,
                                                        mean=default_mean, std=default_std)
    else:
        logger.info('Data augmentation %s is not supported yet.' % (opt.data_aug))

    if opt.dataset == 'kinetics400':
        train_dataset = Kinetics400(setting=opt.train_list, root=data_dir, train=True,
                                    new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                    target_width=input_size, target_height=input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                                    slowfast=opt.slowfast, slow_temporal_stride=opt.slow_temporal_stride, fast_temporal_stride=opt.fast_temporal_stride,
                                    data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_train)
        val_dataset = Kinetics400(setting=opt.val_list, root=val_data_dir, train=False,
                                  new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                  target_width=input_size, target_height=input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                                  slowfast=opt.slowfast, slow_temporal_stride=opt.slow_temporal_stride, fast_temporal_stride=opt.fast_temporal_stride,
                                  data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_test)
    elif opt.dataset == 'ucf101':
        train_dataset = UCF101(setting=opt.train_list, root=data_dir, train=True,
                               new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,
                               target_width=input_size, target_height=input_size,
                               data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_train)
        val_dataset = UCF101(setting=opt.val_list, root=data_dir, train=False,
                             new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,
                             target_width=input_size, target_height=input_size,
                             data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_test)
    elif opt.dataset == 'somethingsomethingv2':
        train_dataset = SomethingSomethingV2(setting=opt.train_list, root=data_dir, train=True,
                                             new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                             target_width=input_size, target_height=input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                                             data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_train)
        val_dataset = SomethingSomethingV2(setting=opt.val_list, root=data_dir, train=False,
                                           new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                           target_width=input_size, target_height=input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                                           data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_test)
    elif opt.dataset == 'hmdb51':
        train_dataset = HMDB51(setting=opt.train_list, root=data_dir, train=True,
                               new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                               target_width=input_size, target_height=input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                               data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_train)
        val_dataset = HMDB51(setting=opt.val_list, root=data_dir, train=False,
                             new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                             target_width=input_size, target_height=input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                             data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_test)
    elif opt.dataset == 'custom':
        # transform_train = video.VideoGroupTrainTransform(size=(224, 224), scale_ratios=[1.0, 0.8], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transform_test = transform_train
        train_dataset = VideoClsCustom(root=data_dir,
                               setting=opt.train_list,
                               train=True,
                               new_length=32,
                                name_pattern='frame_%d.jpg',
                               transform=transform_train,
                               video_loader=False,
                               slowfast = True,
                               use_decord=True,)
        val_dataset = VideoClsCustom(root=val_data_dir,
                               setting=opt.val_list,
                               train=True,
                               new_length=32,
                                name_pattern='frame_%d.jpg',
                               transform=transform_test,
                               video_loader=False,
                               slowfast = True,
                               use_decord=True,)

        # train_dataset = VideoClsCustom(setting=opt.train_list, root=data_dir, train=True,name_pattern ='frame_%d.jpg',
        #                                 video_loader=opt.video_loader, use_decord=opt.use_decord,
        #                                slowfast=opt.slowfast, slow_temporal_stride=opt.slow_temporal_stride, fast_temporal_stride=opt.fast_temporal_stride,
        #                                 num_segments=opt.num_segments, transform=transform_train)
        # val_dataset = VideoClsCustom(setting=opt.val_list, root=val_data_dir, train=False,name_pattern ='frame_%d.jpg',
        #                                 video_loader=opt.video_loader, use_decord=opt.use_decord,
        #                              slowfast=opt.slowfast, slow_temporal_stride=opt.slow_temporal_stride, fast_temporal_stride=opt.fast_temporal_stride,
        #                              num_segments=opt.num_segments, transform=transform_test)
    else:
        logger.info('Dataset %s is not supported yet.' % (opt.dataset))

    logger.info('Load %d training samples and %d validation samples.' % (len(train_dataset), len(val_dataset)))

    

    train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=False, num_workers=num_workers)                         
   
    return train_data, val_data, batch_fn

def main():
    opt = parse_args(parser)

    assert not(os.path.isdir(opt.save_dir)), "already done this experiment..."
    Path(opt.save_dir).mkdir(parents = True)
    
    filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)

    sw = SummaryWriter(logdir=opt.save_dir, flush_secs=5, verbose=False)

   
    if opt.use_amp:
        amp.init()

    batch_size = opt.batch_size
    classes = opt.num_classes

    # num_gpus = opt.num_gpus
    # batch_size *= max(1, num_gpus)
    # logger.info('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))
    # context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    # num_workers = opt.num_workers

    num_gpus = 1
    context = [mx.gpu(i) for i in range(num_gpus)]
    per_device_batch_size = 5
    num_workers = 12
    batch_size = per_device_batch_size * num_gpus

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]

    if opt.slowfast:
        optimizer = 'nag'
    else:
        optimizer = 'sgd'

    if opt.clip_grad > 0:
        optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'clip_gradient': opt.clip_grad}
    else:
        # optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}
        optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum}

    if opt.dtype != 'float32':
        optimizer_params['multi_precision'] = True

    model_name = opt.model
    if opt.use_pretrained and len(opt.hashtag) > 0:
        opt.use_pretrained = opt.hashtag
    net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained,
                    use_tsn=opt.use_tsn, num_segments=opt.num_segments, partial_bn=opt.partial_bn,
                    bn_frozen=opt.freeze_bn)
    # net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    logger.info(net)

    resume_params  = find_model_params(opt)
    if resume_params is not '':
        net.load_parameters(resume_params, ctx=context)
        print('Continue training from model %s.' % (resume_params))

  
    train_data, val_data, batch_fn = get_data_loader(opt, batch_size, num_workers, logger)


    iterations_per_epoch = len(train_data) // opt.accumulate
    lr_scheduler = CyclicalSchedule(CosineAnnealingSchedule, min_lr=0, max_lr=opt.lr,
                            cycle_length=opt.T_0*iterations_per_epoch, cycle_length_decay=opt.T_mult, cycle_magnitude_decay=1)
    optimizer_params['lr_scheduler'] = lr_scheduler

    optimizer  = mx.optimizer.SGD(**optimizer_params)
    train_metric = mx.metric.Accuracy()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    def test(ctx, val_data, kvstore="None"):
        acc_top1.reset()
        acc_top5.reset()
        L = gluon.loss.SoftmaxCrossEntropyLoss()
        num_test_iter = len(val_data)
        val_loss_epoch = 0
        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = []
            for _, X in enumerate(data):
                X = X.reshape((-1,) + X.shape[2:])
                pred = net(X.astype(opt.dtype, copy=False))
                outputs.append(pred)

            loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]

            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

            val_loss_epoch += sum([l.mean().asscalar() for l in loss]) / len(loss)

            if opt.log_interval and not (i+1) % opt.log_interval:
                _, top1 = acc_top1.get()
                _, top5 = acc_top5.get()
                logger.info('Batch [%04d]/[%04d]: acc-top1=%f acc-top5=%f' % (i, num_test_iter, top1*100, top5*100))

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        val_loss = val_loss_epoch / num_test_iter



        return (top1, top5, val_loss)

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]

        if opt.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        if opt.partial_bn:
            train_patterns = "None"
            if 'inceptionv3' in opt.model:
                train_patterns = '.*weight|.*bias|inception30_batchnorm0_gamma|inception30_batchnorm0_beta|inception30_batchnorm0_running_mean|inception30_batchnorm0_running_var'
            elif 'inceptionv1' in opt.model:
                train_patterns = '.*weight|.*bias|googlenet0_batchnorm0_gamma|googlenet0_batchnorm0_beta|googlenet0_batchnorm0_running_mean|googlenet0_batchnorm0_running_var'
            else:
                logger.info('Current model does not support partial batch normalization.')
            
            # trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, optimizer_params, update_on_kvstore=False)
            trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, update_on_kvstore=False)


        elif opt.freeze_bn:
            train_patterns = '.*weight|.*bias'
            # trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, optimizer_params, update_on_kvstore=False)
            trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, update_on_kvstore=False)

        else:
            # trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, update_on_kvstore=False)
            trainer = gluon.Trainer(net.collect_params(), optimizer, update_on_kvstore=False)


        if opt.accumulate > 1:
            params = [p for p in net.collect_params().values() if p.grad_req != 'null']
            for p in params:
                p.grad_req = 'add'

        if opt.resume_states is not '':
            trainer.load_states(opt.resume_states)

        if opt.use_amp:
            amp.init_trainer(trainer)

        L = gluon.loss.SoftmaxCrossEntropyLoss()

        best_val_score = 0
        lr_decay_count = 0

        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()
            train_metric.reset()
            btic = time.time()
            num_train_iter = len(train_data)
            train_loss_epoch = 0
            train_loss_iter = 0

            for i, batch in tqdm(enumerate(train_data)):
                data, label = batch_fn(batch, ctx)

                with ag.record():
                    outputs = []
                    for _, X in enumerate(data):
                        X = X.reshape((-1,) + X.shape[2:])
                        # pred = net(X.astype(opt.dtype, copy=False))
                        pred  = net(X)
                        outputs.append(pred)
                    loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]

                    if opt.use_amp:
                        with amp.scale_loss(loss, trainer) as scaled_loss:
                            ag.backward(scaled_loss)
                    else:
                        ag.backward(loss)

                if opt.accumulate > 1:
                    if (i + 1) % opt.accumulate == 0:
                        trainer.step(batch_size * opt.accumulate)
                        net.collect_params().zero_grad()
                else:
                    trainer.step(batch_size)

                train_metric.update(label, outputs)
                train_loss_iter = sum([l.mean().asscalar() for l in loss]) / len(loss)
                train_loss_epoch += train_loss_iter

                train_metric_name, train_metric_score = train_metric.get()
                sw.add_scalar(tag='train_acc_top1_iter', value=train_metric_score*100, global_step=epoch * num_train_iter + i)
                sw.add_scalar(tag='train_loss_iter', value=train_loss_iter, global_step=epoch * num_train_iter + i)
                sw.add_scalar(tag='learning_rate_iter', value=trainer.learning_rate, global_step=epoch * num_train_iter + i)

                if opt.log_interval and not (i+1) % opt.log_interval:
                    logger.info('Epoch[%03d] Batch [%04d]/[%04d]\tSpeed: %f samples/sec\t %s=%f\t loss=%f\t lr=%f' % (
                                epoch, i, num_train_iter, batch_size*opt.log_interval/(time.time()-btic),
                                train_metric_name, train_metric_score*100, train_loss_epoch/(i+1), trainer.learning_rate))
                    btic = time.time()

            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(batch_size * i /(time.time() - tic))
            mx.ndarray.waitall()

            logger.info('[Epoch %03d] training: %s=%f\t loss=%f' % (epoch, train_metric_name, train_metric_score*100, train_loss_epoch/num_train_iter))
            logger.info('[Epoch %03d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time()-tic))
            sw.add_scalar(tag='train_loss_epoch', value=train_loss_epoch/num_train_iter, global_step=epoch)

            if not opt.train_only:
                acc_top1_val, acc_top5_val, loss_val = test(ctx, val_data)

                logger.info('[Epoch %03d] validation: acc-top1=%f acc-top5=%f loss=%f' % (epoch, acc_top1_val*100, acc_top5_val*100, loss_val))
                sw.add_scalar(tag='val_loss_epoch', value=loss_val, global_step=epoch)
                sw.add_scalar(tag='val_acc_top1_epoch', value=acc_top1_val*100, global_step=epoch)

                if acc_top1_val > best_val_score:
                    best_val_score = acc_top1_val
                    net.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, model_name, epoch))
                    trainer.save_states('%s/%.4f-%s-%s-%03d-best.states'%(opt.save_dir, best_val_score, opt.dataset, model_name, epoch))
                else:
                    if opt.save_frequency and opt.save_dir and (epoch + 1) % opt.save_frequency == 0:
                        net.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, model_name, epoch))
                        trainer.save_states('%s/%s-%s-%03d.states'%(opt.save_dir, opt.dataset, model_name, epoch))

        # save the last model
        net.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, model_name, opt.num_epochs-1))
        trainer.save_states('%s/%s-%s-%03d.states'%(opt.save_dir, opt.dataset, model_name, opt.num_epochs-1))

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)

    train(context)
    sw.close()

if __name__ == '__main__':
    main()
