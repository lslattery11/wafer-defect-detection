import os
import argparse
from datetime import datetime

import wandb
from torch.optim import Adam

from wdd.model.cnn_spp import CNN_SPP_Net,make_spp_training_net
from wdd.model.model_training import train_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate", type=float,
        required = True,
        help='learning rate for optimizer')
    parser.add_argument(
        "--weight_decay", type=float,
        required = True,
        help='weight decay for optimizer')
    parser.add_argument(
        "--batch_step_size", type=int,
        required = True,
        help='batch_step_size for optimizer')
    parser.add_argument(
        "--num_cnn_layers", type=int,
        required = True,
        help='number of cnn layers')
    parser.add_argument(
        "--num_spp_outputs", type=int,
        required = True,
        help='number of spp output channels')
    parser.add_argument(
        "--num_linear_layers", type=int,
        required = True,
        help='number of linear layers')
    parser.add_argument(
        "--transform_prob_threshold", type = float,
        required = True,
        help = "probability threshold for transform")
    parser.add_argument(
        "--epochs", type = int,
        required = True,
        help = "number of epochs.")
    parser.add_argument(
        "--name", type=str,
        required=True,
        help = "base name for wandb logging")
    parser.add_argument(
        "--outpath", type=str,
        required=True,
        help = "outpath for wandb logging")

    args = parser.parse_args()

    cnn_channels=tuple(2**(i) for i in range(args.num_cnn_layers))
    spp_output_sizes=[(1+2*i,1+2*i) for i in range(args.num_spp_outputs)]
    linear_output_sizes=tuple(9*2**(i-1) for i in range(args.num_linear_layers,0,-1))

    model_parameters=dict(
        cnn_channels=cnn_channels,
        spp_output_sizes=spp_output_sizes,
        linear_output_sizes=linear_output_sizes,
    )

    config=dict(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_step_size=args.batch_step_size,
        num_cnn_layers=args.num_cnn_layers,
        num_spp_outputs=args.num_spp_outputs,
        num_linear_layers=args.num_linear_layers,
        transform_prob_threshold=args.transform_prob_threshold,
        epochs=args.epochs,
        model_parameters=model_parameters,
        )
    
    net=make_spp_training_net(config)

    learning_rate=config['learning_rate']
    weight_decay=config['weight_decay']
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #init wandb logging
    outpath=args.outpath

    name=f'{args.name}_{datetime.now().isoformat()}'

    wandb.init(
        project='wafer-defect-detection',
        config=config,
        name=name,
        dir=outpath,
        )

    wandb.watch(
        net,
        log='all',
        log_freq=10,
        )

    wandb.define_metric("training_loss", summary="min")
    wandb.define_metric("validation_loss", summary="min")
    wandb.define_metric("balanced_f1", summary="max")

    train_model(
        net,
        optimizer,
        epochs=args.epochs,
        log=True,
        )
    
    net.save(os.path.join(wandb.run.dir, name+'.pt'))

    wandb.finish()

