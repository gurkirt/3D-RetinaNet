

import time
import datetime
import torch
from modules import utils
import torch.utils.data as data_utils
from modules import  AverageMeter
from data import custum_collate
from modules.solver import get_optim
from val import validate

def train(args, net, train_dataset, val_dataset):
    
    optimizer, scheduler, solver_print_str = get_optim(args, net)

    if args.tensorboard:
        from tensorboardX import SummaryWriter

    source_dir = args.save_root+'/source/' # where to save the source
    utils.copy_source(source_dir)

    print('\nLoading Datasets')

    args.start_iteration = 0
    if args.resume>100:
        args.start_iteration = args.resume
        args.iteration = args.start_iteration
        for _ in range(args.iteration-1):
            scheduler.step()
        model_file_name = '{:s}/model_{:06d}.pth'.format(args.save_root, args.start_iteration)
        optimizer_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.save_root, args.start_iteration)
        net.load_state_dict(torch.load(model_file_name))
        optimizer.load_state_dict(torch.load(optimizer_file_name))
        
    # anchors = anchors.cuda(0, non_blocking=True)
    if args.tensorboard:
        log_dir = args.save_root+'tensorboard-{date:%m-%d-%Hx}.log'.format(date=datetime.datetime.now())
        sw = SummaryWriter(log_dir=log_dir)
    log_file = open(args.save_root+'training.text{date:%m-%d-%Hx}.txt'.format(date=datetime.datetime.now()), 'w', 1)
    log_file.write(args.exp_name+'\n')

    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')
    log_file.write(str(net))
    log_file.write(solver_print_str)
    net.train()
    

    # loss counters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()

    # train_dataset = DetectionDatasetDatasetDatasetDatasetDataset(args, 'train', BaseTransform(args.input_dim, args.means, args.stds))

    log_file.write(train_dataset.print_str)
    log_file.write(val_dataset.print_str)
    print('Train-DATA :::>>>', train_dataset.print_str)
    print('VAL-DATA :::>>>', val_dataset.print_str)
    epoch_size = len(train_dataset) // args.batch_size
    print('Training FPN on ', train_dataset.dataset,'\n')


    train_data_loader = data_utils.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True, collate_fn=custum_collate, drop_last=True)
    
    
    val_data_loader = data_utils.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)
  
    torch.cuda.synchronize()
    start = time.perf_counter()
    iteration = args.start_iteration
    eopch = 0
    num_bpe = len(train_data_loader)
    while iteration <= args.max_iter:
        for i, (images, counts, gt_boxes, gt_labels, img_indexs, wh) in enumerate(train_data_loader):
            if iteration > args.max_iter:
                break
            iteration += 1
            epoch = int(iteration/num_bpe)
            images = images.cuda(0, non_blocking=True)
            gt_boxes = gt_boxes.cuda(0, non_blocking=True)
            gt_labels = gt_labels.cuda(0, non_blocking=True)
            counts = counts.cuda(0, non_blocking=True)
            # forward
            torch.cuda.synchronize()
            data_time.update(time.perf_counter() - start)

            # print(images.size(), anchors.size())
            optimizer.zero_grad()
            # pdb.set_trace()
            # print(gts.shape, counts.shape, images.shape)
            loss_l, loss_c = net(images, gt_boxes, gt_labels, counts, img_indexs)
            loss_l, loss_c = loss_l.mean() , loss_c.mean()
            loss = loss_l + loss_c

            loss.backward()
            optimizer.step()
            scheduler.step()

            # pdb.set_trace()
            loc_loss = loss_l.item()
            conf_loss = loss_c.item()
            # pdb.set_trace()
            # print(pppppppppppp)
            if loc_loss>300:
                lline = '\n\n\n We got faulty LOCATION loss {} {} \n\n\n'.format(loc_loss, conf_loss)
                log_file.write(lline)
                print(lline)
                loc_loss = 20.0
            if conf_loss>300:
                lline = '\n\n\n We got faulty CLASSIFICATION loss {} {} \n\n\n'.format(loc_loss, conf_loss)
                log_file.write(lline)
                print(lline)
                conf_loss = 20.0
            
            # print('Loss data type ',type(loc_loss))
            loc_losses.update(loc_loss)
            cls_losses.update(conf_loss)
            losses.update((loc_loss + conf_loss)/2.0)

            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter()

            if iteration % args.log_step == 0 and iteration > args.log_start:
                if args.tensorboard:
                    sw.add_scalars('Classification', {'val': cls_losses.val, 'avg':cls_losses.avg},iteration)
                    sw.add_scalars('Localisation', {'val': loc_losses.val, 'avg':loc_losses.avg},iteration)
                    sw.add_scalars('Overall', {'val': losses.val, 'avg':losses.avg},iteration)
                    
                print_line = 'Itration [{:d}]{:06d}/{:06d} loc-loss {:.2f}({:.2f}) cls-loss {:.2f}({:.2f}) ' \
                             'average-loss {:.2f}({:.2f}) DataTime{:0.2f}({:0.2f}) Timer {:0.2f}({:0.2f})'.format( epoch,
                              iteration, args.max_iter, loc_losses.val, loc_losses.avg, cls_losses.val,
                              cls_losses.avg, losses.val, losses.avg, 10*data_time.val, 10*data_time.avg, 10*batch_time.val, 10*batch_time.avg)

                log_file.write(print_line+'\n')
                print(print_line)
                if iteration % (args.log_step*10) == 0:
                    print_line = args.exp_name
                    log_file.write(print_line+'\n')
                    print(print_line)


            if (iteration % args.val_step == 0 or iteration== args.intial_val or iteration == args.max_iter) and iteration>0:
                torch.cuda.synchronize()
                tvs = time.perf_counter()
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), '{:s}/model_{:06d}.pth'.format(args.save_root, iteration))
                torch.save(optimizer.state_dict(), '{:s}/optimizer_{:06d}.pth'.format(args.save_root, iteration))
                net.eval() # switch net to evaluation mode
                mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, iteration)
                
                net.train()
                if args.fbn:
                    if args.multi_gpu:
                        net.module.backbone_net.apply(utils.set_bn_eval)
                    else:
                        net.backbone_net.apply(utils.set_bn_eval)

                for nlt in range(args.nlts):
                    for ap_str in ap_strs[nlt]:
                        print(ap_str)
                        log_file.write(ap_str+'\n')
                    ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(args.label_types[nlt], mAP[nlt])
                    print(ptr_str)
                    log_file.write(ptr_str)

                    if args.tensorboard:
                        sw.add_scalar('{:s}mAP'.format(args.label_types[nlt]), mAP[nlt], iteration)
                        class_AP_group = dict()
                        for c, ap in enumerate(ap_all[nlt]):
                            class_AP_group[all_classes[nlt][c]] = ap
                        sw.add_scalars('ClassAP-{:s}'.format(args.label_types[nlt]), class_AP_group, iteration)


                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
                print(prt_str)
                log_file.write(ptr_str)

    log_file.close()

# if __name__ == '__main__':
#     main()
