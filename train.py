"""
 Main script for training 
 - Mostly inherited from Stanford CS230 example code:
   https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision

"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate

# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir'    , default='datasets/spec_split'   , help="Directory containing the splitted dataset")
parser.add_argument('--model_dir'   , default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--num_classes' , default=300, type=int, help="Numer of classes as in splitting datasets")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")

# writer = SummaryWriter('tensorboardlogs/trainlog')

def train(model, optimizer, loss_fn, dataloader, metrics, params, epoch, logger):
    """Train the model on `num_steps batches

    Args:
        model: (torch.nn.Module) 
            the neural network
        optimizer: (torch.optim) 
            optimizer for parameters of model
        loss_fn: 
            a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) 
            a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) 
            a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) 
            hyperparameters
        num_steps: (int) 
            number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)

            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch.float(), labels_batch.float())

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch, params.threshold)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

                ## tensorboard logging
                niter = epoch*len(dataloader)+i
                for tag, value in summary_batch.items():
                    logger.scalar_summary(tag, value, niter)
                    # writer.add_scalar(tag, value, niter)

                #-- Log values and gradients of the parameters (histogram)
                # for name, param in model.named_parameters():
                #     name = name.replace('.', '/')
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), niter)
                #     writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), niter)                

            # update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) 
            the neural network
        train_dataloader: (DataLoader) 
            a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) 
            a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) 
            optimizer for parameters of model
        loss_fn: 
            a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) 
            a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) 
            hyperparameters
        model_dir: (string) 
            directory containing config, weights and log
        restore_file: (string) 
            optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_met = 0.0

    ## tensorboard loggers
    logger_train = utils.Logger(model_dir+"tensorboardlogs/train/")
    logger_eval  = utils.Logger(model_dir+"tensorboardlogs/eval/")

    scheduler = None
    if hasattr(params,'lr_decay_gamma'):
        scheduler = StepLR(optimizer, step_size=params.lr_decay_step, gamma=params.lr_decay_gamma)

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        ##
        if scheduler is not None:
            scheduler.step()

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch, logger_train)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, args.num_classes, epoch, logger_eval)

        if hasattr(params,'if_single'): 
            if params.if_single == 1: # single-label
                val_acc = val_metrics['accuracy']
        else:
            val_acc = val_metrics['f1']

        is_best = val_acc >= best_val_met

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_met = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(1240)
    if params.cuda: torch.cuda.manual_seed(1240)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    if hasattr(params,'if_single'): 
        if params.if_single == 1: # single-label
            dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params, mixing=False)
    else:
        dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params, mixing=True)

    train_dl = dataloaders['train']
    val_dl   = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    if params.model == 1:
        print('  -- Training using DenseNet')
        model = net.DenseNetBase(params,args.num_classes).cuda() if params.cuda else net.DenseNetBase(params,args.num_classes)

    elif params.model == 2:
        print('  -- Training using SqueezeNet')
        model = net.SqueezeNetBase(params,args.num_classes).cuda() if params.cuda else net.SqueezeNetBase(params,args.num_classes)

    elif params.model == 3:
        print('  -- Training using Inception')
        model = net.InceptionBase(params,args.num_classes).cuda() if params.cuda else net.InceptionBase(params,args.num_classes)

    elif params.model == 4:
        print('  -- Training using InceptionResNet')
        model = net.InceptionResnetBase(params,args.num_classes).cuda() if params.cuda else net.InceptionResnetBase(params,args.num_classes)

    elif params.model == 5:
        print('  -- Training using ResNet')
        model = net.ResNet14(params,args.num_classes).cuda() if params.cuda else net.ResNet14(params,args.num_classes)

    elif params.model == 6:
        print('  -- Training using DenseNet with Binary Relevance')
        model = net.DenseBR(params,args.num_classes).cuda() if params.cuda else net.DenseBR(params,args.num_classes)

    elif params.model == 7:
        print('  -- Training using ResNet with Binary Relevance')
        model = net.ResBR(params,args.num_classes).cuda() if params.cuda else net.ResBR(params,args.num_classes)

    elif params.model == 8:
        print('  -- Training using DenseNet + BLSTM')
        model = net.DenseNetBLSTM(params,args.num_classes).cuda() if params.cuda else net.DenseNetBLSTM(params,args.num_classes)


    # optimizer
    if params.optimizer == 1:
        print('  ---optimizer is Adam')
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        if hasattr(params,'lambd'):
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.lambd)
    elif params.optimizer == 2:
        print('  ---optimizer is SGD')
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=params.learning_rate)
        if hasattr(params,'lambd'):
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.lambd)

    ## tensorboar logging
    dummy_input = Variable(torch.rand(params.batch_size,1,128,params.width).cuda(async=True) if params.cuda else \
                           torch.rand(params.batch_size,1,128,params.width))
    # writer.add_graph(model, (dummy_input, ))

    # fetch loss function and metrics
    if hasattr(params,'if_single'): 
        if params.if_single == 1: # single-label
            loss_fn = net.loss_fn_sing
            metrics = net.metrics_sing
    else:
        if hasattr(params,'loss_fn'):
            if params.loss_fn == 1: # use WARP loss
                print('  ---loss function is WARP'); print('')
                loss_fn = net.loss_warp
            elif params.loss_fn == 2: # use LSEP loss
                print('  ---loss function is LSEP'); print('')
                loss_fn = net.loss_lsep
        else:
            print('  ---loss function is BCE'); print('')
            loss_fn = net.loss_fn
        metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)

    # writer.export_scalars_to_json("./all_scalars.json")
    # writer.close()
