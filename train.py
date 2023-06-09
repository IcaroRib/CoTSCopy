import argparse
import os
import time
import datetime
import math
import numpy as np
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

# import methodsF
from cost import CoST


def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--archive', type=str, required=True, help='The archive name that the dataset belongs to. This can be set to forecast_csv, or forecast_csv_univar')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--recurrent', type=str, default=None, help='Define a type of recurrent encoder. Options are rnn, lstm and gru')
    parser.add_argument('--attention', type=str, default=None, help='Define a type of attention mechanism. Options are transformer')
    parser.add_argument('--conv', type=str, default=None, help='Define a type of convolutional mechanism. Options are tcn')

    parser.add_argument('--kernels', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128], help='The kernel sizes used in the mixture of AR expert layers')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Weighting hyperparameter for loss function')

    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    if args.archive == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        test_data = data[:, test_slice]
    elif args.archive == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        test_data = data[:, test_slice]
    elif args.archive == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        test_data = data[:, test_slice]
    elif args.archive == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        test_data = data[:, test_slice]
    else:
        raise ValueError(f"Archive type {args.archive} is not supported.")

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        attention=args.attention,
        recurrent=args.recurrent,
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)


    file_prefix = "conv"
    if args.recurrent:
        file_prefix = args.recurrent
    elif args.attention:
        file_prefix = args.attention
    elif args.conv:
        file_prefix = args.conv

    run_dir = f"training/{args.dataset}/{name_with_datetime(args.run_name, args.alpha, file_prefix)}"

    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()

    model = CoST(
        input_dims=train_data.shape[-1],
        kernels=args.kernels,
        alpha=args.alpha,
        max_train_length=args.max_train_length,
        device=device,
        **config
    )

    loss_log, eval_loss = model.fit(
        train_data,
        test_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')
    pkl_save(f'{run_dir}/train_loss.pkl', loss_log)
    pkl_save(f'{run_dir}/eval_loss.pkl', eval_loss)

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, args.max_train_length-1)
        print('Evaluation result:', eval_res)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        pkl_save(f'{run_dir}/out.pkl', out)

    print("Finished.")
