import argparse, time, logging, os, sys, math
import gc

import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from mxnet import gluon, nd, gpu, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import quantize_net

from gluoncv.data.transforms import video
from gluoncv.data import UCF101, Kinetics400, SomethingSomethingV2, HMDB51,VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load
from parser_helper import parse_args, find_model_params

import pickle
from pathlib import Path
from Evaluation_video import Evaluator_video
import pandas as pd
import joblib
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score)


parser = argparse.ArgumentParser()

parser.add_argument("--default_path", type=str,
                    help="File with defautl config settings for testing")
                    
parser.add_argument("--custom_path", type=str,
                    help="File with current custom config settings for testing")

def batch_fn(batch, ctx):
    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label

def _list_to_numpy(array):
    return np.concatenate([batch_values[0].asnumpy() for batch_values in array])


def test(ctx, val_data, opt, net):
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    true_labels = [] 
    predictions = []

    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = []
        for _, X in enumerate(data):
            X = X.reshape((-1,) + X.shape[2:])
            # pred = net(X.astype(opt.dtype, copy=False))
            pred  = net(X)
            if opt.use_softmax:
                pred = F.softmax(pred, axis=1)
            outputs.append(pred)

        predictions.append(outputs)
        true_labels.append(label)
    
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
        mx.ndarray.waitall()

        _, cur_top1 = acc_top1.get()
        _, cur_top5 = acc_top5.get()

        if i > 0 and i % opt.log_interval == 0:
            print('%04d/%04d is done: acc-top1=%f acc-top5=%f' % (i, len(val_data), cur_top1*100, cur_top5*100))
   
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    #save true_labels, predictions
    predictions  = _list_to_numpy(predictions)
    true_labels  = _list_to_numpy(true_labels)
    np.save(os.path.join(opt.save_dir,"labels"),true_labels)
    np.save(os.path.join(opt.save_dir,"predictions"),predictions)
  
    return top1,top5, true_labels,predictions

def get_split_report(evaluator,split):
    thresholds  = np.linspace(0,1,num = 15)    
    #init metrics dict
    metrics = evaluator.init_metrics()
    for count,thr in enumerate(thresholds):
        true_labels, predicted_labels  = evaluator.filter_predictions(thr)
        evaluator.plot_confusion_matrix(true_labels,predicted_labels,title= "Confusion_Matrix",epoch=count)
        metrics = evaluator._update_test_metrics(true_labels,predicted_labels,metrics)
        #compute class Precision,Recall
        evaluator.plot_class_performance(metrics["Precision_Class"][count],"Precision",count,title= "Class_Precision")
        evaluator.plot_class_performance(metrics["Recall_Class"][count],"Recall",count,title= "Class_Recall")
    
    #plot metrics evolution
    metrics["threshold"] = thresholds
    #mcc
    evaluator.plot_metric("Mcc","threshold",metrics,"MCC curve")
    metrics["Mcc_auc"] = auc(metrics["threshold"],metrics["Mcc"])
    #coverages
    evaluator.plot_metric("coverages","threshold",metrics,"Coverage curve")
    metrics["coverages_auc"] = auc(metrics["threshold"],metrics["coverages"])
    #Pr curves
    evaluator.plot_metric("Recall_Avg","Precision_Avg",metrics,"Pr curve")
    metrics["pr_auc"] = auc(metrics["Recall_Avg"],metrics["Precision_Avg"])

    #save only avg metrics
    avg_metrics = {metric: value for metric,value in metrics.items() if not("Class" in metric)}
    return avg_metrics

def save_results(metrics,split_folder):
    table_file = os.path.join(split_folder,"metrics.csv")
    metrics_df  = pd.DataFrame.from_dict(metrics)
    metrics_df.to_csv(table_file,index=False)

def benchmarking(opt, net, ctx):
    bs = opt.batch_size
    num_iterations = opt.num_iterations
    input_size = opt.input_size
    size = num_iterations * bs
    input_shape = (bs * opt.num_segments, 3, opt.new_length, input_size, input_size)
    data = mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=ctx[0], dtype='float32')
    if opt.new_length == 1:
        # this is for 2D input case
        data = nd.squeeze(data, axis=2)
    dry_run = 5

    from tqdm import tqdm
    with tqdm(total=size + dry_run * bs) as pbar:
        for n in range(dry_run + num_iterations):
            if n == dry_run:
                tic = time.time()
            output = net(data)
            output.wait_to_read()
            pbar.update(bs)
    speed = size / (time.time() - tic)
    print('With batch size %d , %d batches, throughput is %f imgs/sec' % (bs, num_iterations, speed))


def calibration(net, val_data, opt, ctx, logger):
    if isinstance(ctx, list):
        ctx = ctx[0]
    ctx  = mx.cpu()
    exclude_sym_layer = []
    exclude_match_layer = []
    if 'inceptionv3' not in opt.model:
        exclude_match_layer += ['concat']
    if opt.num_gpus > 0:
        raise ValueError('currently only supports CPU with MKL-DNN backend')
    net = quantize_net(net, calib_data=val_data, quantized_dtype=opt.quantized_dtype, calib_mode=opt.calib_mode,
                       exclude_layers=exclude_sym_layer, num_calib_examples=opt.batch_size * opt.num_calib_batches,
                       exclude_layers_match=exclude_match_layer, ctx=ctx, logger=logger)
    # net = quantize_net(net, calib_data=val_data, quantized_dtype=opt.quantized_dtype, quantize_mode='full', calib_mode=opt.calib_mode,
    #                    exclude_layers=exclude_sym_layer, num_calib_examples=opt.batch_size * opt.num_calib_batches,
    #                    exclude_layers_match=exclude_match_layer, ctx=ctx, logger=logger)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dst_dir = os.path.join(dir_path, 'model')
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    prefix = os.path.join(dst_dir, opt.model + '-quantized-' + opt.calib_mode)
    logger.info('Saving quantized model at %s' % dst_dir)
    net.export(prefix, epoch=0)


def main(logger):
    opt = parse_args(parser)
    print(opt)
    
    assert not(os.path.isdir(opt.save_dir)), "already done this experiment..."
    Path(opt.save_dir).mkdir(parents = True)
    # Garbage collection, default threshold is (700, 10, 10).
    # Set threshold lower to collect garbage more frequently and release more CPU memory for heavy data loading.
    gc.set_threshold(100, 5, 5)


    num_gpus = 1
    context = [mx.gpu(i) for i in range(num_gpus)]
    per_device_batch_size = 5
    num_workers = 12
    batch_size = per_device_batch_size * num_gpus
    num_workers = opt.num_workers

    print('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))

    # get data
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    # if opt.ten_crop:
    #     if opt.data_aug == 'v1':
    #         transform_test = transforms.Compose([
    #             video.VideoTenCrop(opt.input_size),
    #             video.VideoToTensor(),
    #             video.VideoNormalize(default_mean, default_std)
    #         ])
    #     else:
    #         transform_test = transforms.Compose([
    #             video.ShortSideRescale(opt.input_size),
    #             video.VideoTenCrop(opt.input_size),
    #             video.VideoToTensor(),
    #             video.VideoNormalize(default_mean, default_std)
    #         ])
    #     opt.num_crop = 10
    # elif opt.three_crop:
    #     if opt.data_aug == 'v1':
    #         transform_test = transforms.Compose([
    #             video.VideoThreeCrop(opt.input_size),
    #             video.VideoToTensor(),
    #             video.VideoNormalize(default_mean, default_std)
    #         ])
    #     else:
    #         transform_test = transforms.Compose([
    #             video.ShortSideRescale(opt.input_size),
    #             video.VideoThreeCrop(opt.input_size),
    #             video.VideoToTensor(),
    #             video.VideoNormalize(default_mean, default_std)
    #         ])
    #     opt.num_crop = 3
    # else:
    #     if opt.data_aug == 'v1':
    #         transform_test = video.VideoGroupValTransform(size=opt.input_size, mean=default_mean, std=default_std)
    #     else:
    #         transform_test = video.VideoGroupValTransformV2(crop_size=(opt.input_size, opt.input_size), short_side=opt.input_size,
    #                                                         mean=default_mean, std=default_std)
    #     opt.num_crop = 1

    if not opt.deploy:
        # get model
        if opt.use_pretrained and len(opt.hashtag) > 0:
            opt.use_pretrained = opt.hashtag
        classes = opt.num_classes
        model_name = opt.model
        # Currently, these is no hashtag for int8 models.
        if opt.quantized:
            model_name += '_int8'
            opt.use_pretrained = True

        net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained, num_segments=opt.num_segments, num_crop=opt.num_crop)
        net.cast(opt.dtype)
        net.collect_params().reset_ctx(context)
        resume_params  = find_model_params(opt)

        if opt.mode == 'hybrid':
            net.hybridize(static_alloc=True, static_shape=True)
        if resume_params is not '' and not opt.use_pretrained:
            net.load_parameters(resume_params, ctx=context)
            print('Pre-trained model %s is successfully loaded.' % (resume_params))
        else:
            print('Pre-trained model is successfully loaded from the model zoo.')
    else:
        model_name = 'deploy'
        net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(opt.model_prefix),
                    ['data'], '{}-0000.params'.format(opt.model_prefix))
        net.hybridize(static_alloc=True, static_shape=True)

    print("Successfully loaded model {}".format(model_name))
    # dummy data for benchmarking performance
    if opt.benchmark:
        benchmarking(opt, net, context)
        sys.exit()

    if opt.dataset == 'ucf101':
        val_dataset = UCF101(setting=opt.val_list, root=opt.data_dir, train=False,
                             new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,
                             target_width=opt.input_size, target_height=opt.input_size,
                             test_mode=True, data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_test)
    elif opt.dataset == 'kinetics400':
        val_dataset = Kinetics400(setting=opt.val_list, root=opt.data_dir, train=False,
                                  new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                  target_width=opt.input_size, target_height=opt.input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                                  slowfast=opt.slowfast, slow_temporal_stride=opt.slow_temporal_stride, fast_temporal_stride=opt.fast_temporal_stride,
                                  test_mode=True, data_aug=opt.data_aug, num_segments=opt.num_segments, num_crop=opt.num_crop, transform=transform_test)
    elif opt.dataset == 'somethingsomethingv2':
        val_dataset = SomethingSomethingV2(setting=opt.val_list, root=opt.data_dir, train=False,
                                           new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                           target_width=opt.input_size, target_height=opt.input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                                           data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_test)
    elif opt.dataset == 'hmdb51':
        val_dataset = HMDB51(setting=opt.val_list, root=opt.data_dir, train=False,
                             new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                             target_width=opt.input_size, target_height=opt.input_size, video_loader=opt.video_loader, use_decord=opt.use_decord,
                             data_aug=opt.data_aug, num_segments=opt.num_segments, transform=transform_test)

    elif opt.dataset == 'custom':
        transform_test = video.VideoGroupTrainTransform(size=(224, 224), scale_ratios=[1.0, 0.8], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        val_dataset = VideoClsCustom(root=opt.val_data_dir,
                               setting=opt.val_list,
                               train=False,
                               new_length=32,
                                name_pattern='frame_%d.jpg',
                               transform=transform_test,
                               video_loader=False,
                               slowfast = True,
                               use_decord=True,)

    else:
        logger.info('Dataset %s is not supported yet.' % (opt.dataset))

    # val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    #                                  prefetch=int(opt.prefetch_ratio * num_workers), last_batch='discard')
    val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=False, num_workers=num_workers)                         

    print('Load %d test samples in %d iterations.' % (len(val_dataset), len(val_data)))

    # calibrate FP32 model into INT8 model
    if opt.calibration:
        calibration(net, val_data, opt, context, logger)
        sys.exit()


    start_time = time.time()
    acc_top1_val, acc_top5_val, true_labels, predicted_probabilities = test(context, val_data, opt, net)
    split_filename  = os.path.split(opt.val_list)[1]
    split = os.path.splitext(split_filename)[0]
    #load encoder
    encoder  = joblib.load(opt.encoder_path)
    #set-up metrics
    classes = np.arange(len(encoder.classes_))
    metrics_dict={ "Accuracy":balanced_accuracy_score,
                "Mcc":matthews_corrcoef,
                "Precision_Avg": [precision_score,{"average":"micro"}],
                "Recall_Avg" : [recall_score,{"average":"micro"}],
                "Precision_Class": [precision_score,{"labels":classes,"average":None}],
                "Recall_Class" : [recall_score,{"labels":classes,"average":None}],
                }
    split_folder = os.path.join(opt.save_dir,split) 
    #set-up evaluator
    evaluator = Evaluator_video(split_folder,
                        encoder,
                        true_labels,
                        predicted_probabilities,
                        metrics_dict)
    #compute report
    report = get_split_report(evaluator)
    #save report
    save_results(report,split_folder)
    print(f"Correctly process split {split}")

    end_time = time.time()

    print('Test accuracy: acc-top1=%f acc-top5=%f' % (acc_top1_val*100, acc_top5_val*100))
    print('Total evaluation time is %4.2f minutes' % ((end_time - start_time) / 60))


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    main(logger)
