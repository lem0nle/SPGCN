import argparse
import pandas as pd


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=str, default='-1',
                           help="GPU, can be a list of gpus for multi-gpu training,"
                                " e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument('-m', default='NCF', help='baseline model')
    args = argparser.parse_args()
    devices = list(map(int, args.gpu.split(',')))

    # load data
    train = pd.read_csv('data/train_df.csv')
    test = pd.read_csv('data/test_df.csv')
    print(f'Data loaded, #user: {len(set(data.src_ind))}, #item: {len(set(data.dst_ind))}, ')

    # data reindex

    # dataloader
    # model engine
    # train & save
