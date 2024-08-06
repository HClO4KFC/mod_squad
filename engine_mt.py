import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
from util.metric import *

def get_loss(outputs, targets, task):
    criterion = torch.nn.CrossEntropyLoss()
    if 'class' in task:
        task_loss = F.mse_loss(outputs, targets.squeeze(1))
    elif 'segment_semantic' in task:
        # task_loss = criterion(outputs.squeeze(-1), targets)
        task_loss = F.cross_entropy(outputs.squeeze(-1), targets)
    elif 'normal' in task:
        T = targets.permute(0,2,3,1)
        task_loss = (1 - (outputs*T).sum(-1) / (torch.norm(outputs, p=2, dim=-1) + 0.000001) / (torch.norm(T, p=2, dim=-1)+ 0.000001) ).mean()
    elif 'depth' in task or 'keypoint' in task or 'reshading' in task or 'edge' in task or 'segment' in task:
        if outputs.shape[-1] == 1:
            Out = outputs.view(outputs.shape[:-1])
        elif outputs.shape[-1] == 3:
            Out = outputs.permute(0,3,1,2)
        task_loss = F.l1_loss(Out, targets)
    else: # L2 curvature
        if outputs.shape[-1] == 1:
            Out = outputs.view(outputs.shape[:-1])
        elif outputs.shape[-1] > 1:
            Out = outputs.permute(0,3,1,2)
        task_loss = F.mse_loss(Out, targets)
    return task_loss

def get_metric(outputs, targets, task):
    # get the metric
    if 'class' in task:
        # correct_prediction = tf.equal(tf.argmax(final_output,1), tf.argmax(target, 1))
        metric = (outputs.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean().item()
    elif 'depth' in task:
        if outputs.shape[-1] == 1:
            outputs = outputs.view(outputs.shape[:-1]) # B, H, W
        if task == 'depth_euclidean':
            metric = compute_depth_errors(outputs, targets).item()
        else:
            metric = 0.0
    elif 'curvature' in task:
        if outputs.shape[-1] == 1:
            outputs = outputs.view(outputs.shape[:-1])
        elif outputs.shape[-1] > 1:
            outputs = outputs.permute(0,3,1,2)
        metric = F.mse_loss(outputs, targets).item()
    else:
        if outputs.shape[-1] == 1:
            outputs = outputs.view(outputs.shape[:-1])
        elif outputs.shape[-1] == 3:
            outputs = outputs.permute(0,3,1,2)
        metric = F.l1_loss(outputs, targets).item()
    return metric


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, AWL=None,
                    args=None):
    model.train(True)
    AWL.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (data) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        
        if data_iter_step % accum_iter == 0:
            if args.cycle:
                lr_sched.adjust_cycle_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            else:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = data['rgb'].to(device, non_blocking=True)
        z_loss = 0
        loss = 0
        the_loss = {}
        loss_list = []
        tot_loss = 0
        the_metric = {}

        if args.visualize and misc.is_main_process():
            if data_iter_step > 0 and data_iter_step%20 == 0:
                model.module.visualize(vis_head=True, vis_mlp=False, model_name=args.exp_name)

        with torch.cuda.amp.autocast():
            predict = {}
            if model.module.taskGating:
                outputs, aux_loss = model(samples, None)
                z_loss = z_loss + aux_loss
                for task in args.img_types:
                    if 'rgb' in task:
                        continue
                    predict[task] = outputs[task].detach().cpu()
                    targets = data[task].to(device, non_blocking=True)
                    task_loss = get_loss(outputs[task], targets, task)

                    if not math.isfinite(task_loss.item()):
                        print("Loss is {}, stopping training".format(task_loss.item()))
                        sys.exit(1)

                    task_loss = torch.clamp(task_loss, min=-1000, max=1000)
                    tot_loss = tot_loss + task_loss.item()
                    the_loss[task] = task_loss
                    loss_list.append(the_loss[task])
                    task_metric = get_metric(outputs[task], targets, task)
                    the_metric[task] = task_metric
            else:
                
                for task in args.img_types:
                    if 'rgb' in task:
                        continue
                    outputs, aux_loss = model(samples, task)
                    z_loss = z_loss + aux_loss
                    predict[task] = outputs.detach().cpu()

                    targets = data[task].to(device, non_blocking=True)
                    task_loss = get_loss(outputs, targets, task)

                    task_loss = torch.clamp(task_loss, min=-1000, max=1000)
                    task_loss_value = task_loss.item()
                    if not math.isfinite(task_loss_value):
                        print("Task is {} Loss is {}, stopping training".format(task, task_loss_value))
                        task_loss = torch.clamp(task_loss, min=-1000, max=1000)
                        sys.exit(1)

                    tot_loss = tot_loss + task_loss.item()
                    the_loss[task] = task_loss
                    loss_list.append(the_loss[task])
                    task_metric = get_metric(outputs, targets, task)
                    the_metric[task] = task_metric

            if args.visualizeimg:
                image_visualize(args, data, predict)
        

        loss = AWL(loss_list)
        loss_value = loss.item()

        if args.tasks > 2:
            if torch.is_tensor(z_loss):
                if not math.isfinite(z_loss.item()): #
                    print("ZLoss is {}, stopping training".format(z_loss.item()))
                    # z_loss =s
                    sys.exit(1)
            loss = loss + z_loss
        else:
            z_loss = z_loss * 0.00001

        if torch.is_tensor(z_loss):
            z_loss_value = z_loss.item()
        else:
            z_loss_value = z_loss

        the_loss_value = {}
        for _key, value in the_loss.items():
            the_loss_value[_key] = value.item()

        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            loss = torch.clamp(loss, min=-1000, max=1000)
            sys.exit(1)
        loss = torch.clamp(loss, min=-1000, max=1000)
        
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            # model.module.init_aux_statistics()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        if model.module.ismoe:
            metric_logger.update(zloss=z_loss_value)

        for _key, value in the_loss_value.items():
            metric_logger.meters[_key].update(value)

        for _key, value in the_metric.items():
            metric_logger.meters['met_'+_key].update(value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        tot_loss_reduce = misc.all_reduce_mean(tot_loss)
        if torch.is_tensor(z_loss):
            z_loss_value_reduce = misc.all_reduce_mean(z_loss_value)
        else:
            z_loss_value_reduce = 0

        the_loss_value_reduce = {}
        for _key, value in the_loss_value.items():
            the_loss_value_reduce[_key] = misc.all_reduce_mean(value)

        the_metric_reduce = {}
        for _key, value in the_metric.items():
            the_metric_reduce[_key] = misc.all_reduce_mean(value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('z_loss', z_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('tot_loss', tot_loss_reduce, epoch_1000x)
            
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            for _key, value in the_loss_value_reduce.items():
                log_writer.add_scalar('multitask/' + _key, value, epoch_1000x)

            for _key, value in the_metric_reduce.items():
                log_writer.add_scalar('multitask_metric/' + _key, value, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # if misc.is_main_process():
    print("Averaged stats:", metric_logger)
    print('params: ', AWL.params)
    # model.module.visualize(vis_head=False, vis_mlp=True)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, AWL, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    AWL.eval()

    for data in metric_logger.log_every(data_loader, 10, header):
        samples = data['rgb']
        samples = samples.to(device, non_blocking=True)

        the_loss = {}
        loss_list = []
        the_metric = {}
        tot_loss = 0
        with torch.cuda.amp.autocast():
            if model.module.taskGating:
                outputs, _ = model(samples, None)
                # z_loss = z_loss + aux_loss
                for task in args.img_types:
                    if 'rgb' in task:
                        continue
                    targets = data[task].to(device, non_blocking=True)
                    task_loss = get_loss(outputs[task], targets, task)
                    tot_loss = tot_loss + task_loss.item()
                    the_loss[task] = task_loss
                    loss_list.append(the_loss[task])
                    task_metric = get_metric(outputs[task], targets, task)
                    the_metric[task] = task_metric
            else:
                for task in args.img_types:
                    if 'rgb' in task:
                        continue
                    outputs, _ = model(samples, task)
                    targets = data[task].to(device, non_blocking=True)
                    task_loss = get_loss(outputs, targets, task)
                    tot_loss = tot_loss + task_loss.item()
                    the_loss[task] = task_loss
                    loss_list.append(the_loss[task])
                    task_metric = get_metric(outputs, targets, task)
                    the_metric[task] = task_metric

        loss = AWL(loss_list)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(tot_loss=tot_loss)
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        for _key, value in the_loss.items():
            metric_logger.meters[_key].update(value.item(), n=batch_size)

        for _key, value in the_metric.items():
            metric_logger.meters['met_'+_key].update(value, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print('test Result: ', ' '.join(str(a) + ':' + str(b.global_avg) for (a,b) in metric_logger.meters.items()))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
