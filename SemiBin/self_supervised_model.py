import torch
from torch.utils.data import DataLoader
import os
from torch.optim import lr_scheduler
import sys
from .semi_supervised_model import Semi_encoding_single, Semi_encoding_multiple, feature_Dataset
from .utils import norm_abundance

def loss_function(embedding1, embedding2, label):
    # relu = torch.nn.ReLU()
    # d = torch.norm(embedding1 - embedding2, p=2, dim=1)
    # square_pred = torch.square(d)
    # margin_square = torch.square(relu(1 - d))
    # supervised_loss = torch.mean(
    #     label * square_pred + (1 - label) * margin_square)
    # loss = torch.nn.BCELoss()
    # d = torch.exp(-torch.norm(embedding1 - embedding2, p=2, dim=1) / embedding1.shape[1])
    # supervised_loss = loss(d, label)
    relu = torch.nn.ReLU()
    d = torch.norm(embedding1 - embedding2, p=2, dim=1)
    square_pred = torch.square(d)
    margin_square = torch.square(relu(0.1 - d))
    supervised_loss = torch.mean(
        label * square_pred + (1 - label) * margin_square)
    return supervised_loss


def train_self(logger, out : str, datapaths, data_splits, is_combined=True,
          batchsize=2048, epoches=15, device=None, num_process = 8, mode = 'single'):
    """
    Train model from one sample(mode=single) or several samples(mode=several)

    Saves model to disk and returns it

    Parameters
    ----------
    out : filename to write model to
    """
    epoches = 50
    learning_rate = 1e-3
    batchsize = 1024 * 2
    upper_bound = 4_000_000
    from tqdm import tqdm
    import pandas as pd
    import numpy as np

    train_data = pd.read_csv(datapaths[0], index_col=0).values

    if not is_combined:
        train_data = train_data[:, :136]

    torch.set_num_threads(num_process)

    logger.info('Training model...')

    if not is_combined:
        model = Semi_encoding_single(train_data.shape[1])
    else:
        model = Semi_encoding_multiple(train_data.shape[1])

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    loss_list = []
    for epoch in tqdm(range(epoches)):
        for data_index, (datapath, data_split_path) in enumerate(zip(datapaths, data_splits)):
            if epoch == 0:
                logger.debug(f'Reading training data for index {data_index}...')

            data = pd.read_csv(datapath, index_col=0)
            data.index = data.index.astype(str)
            data_split = pd.read_csv(data_split_path, index_col=0)

            if epoch == 0:
                logger.debug(f'Data shape from file `{datapath}`: {data.shape}')
                logger.debug(f'Data shape from file `{data_split_path}` (split data file): {data_split.shape}')

            if mode == 'several':
                if data.shape[1] != 138 or data_split.shape[1] != 136:
                    sys.stderr.write(
                        f"Error: training mode with several only used in single-sample binning!\n")
                    sys.exit(1)

            train_data = data.values
            train_data_split = data_split.values
            n_must_link = len(train_data_split)
            if not is_combined:
                train_data = train_data[:, :136]
            else:
                if norm_abundance(train_data):
                    from sklearn.preprocessing import normalize
                    norm = np.sum(train_data, axis=0)
                    train_data = train_data / norm
                    train_data_split = train_data_split / norm
                    train_data = normalize(train_data, axis=1, norm='l1')
                    train_data_split = normalize(train_data_split, axis=1, norm='l1')


            data_length = len(train_data)
            # cannot link data is sampled randomly
            # n_cannot_link = n_must_link * 1000 // 2 #min(n_must_link * 1000 // 2, 4_000_000)
            n_cannot_link = min(n_must_link * 1000 // 2, upper_bound)
            indices1 = np.random.choice(data_length, size=n_cannot_link)
            indices2 = np.random.choice(data_length,  size=n_cannot_link)


            if epoch == 0:
                logger.debug(
                    f'Number of must-link pairs: {len(train_data_split)//2}')
                logger.debug(
                    f'Number of cannot-link pairs: {n_cannot_link}')

            # I won't use the train_data, instead I'll used train_data_split
            # The initial len(indices1) represents the negative instances and the remaining ones are the positive.
            train_input_1 = np.concatenate((train_data_split[indices1], train_data_split[::2]))
            train_input_2 = np.concatenate((train_data_split[indices2], train_data_split[1::2]))
            train_labels = np.zeros(len(train_input_1), dtype=np.float32)
            train_labels[len(indices1):] = 1

            dataset = feature_Dataset(train_input_1, train_input_2, train_labels)
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=batchsize,
                shuffle=True,
                num_workers=0,
                drop_last=True)

            current_loss = 0
            for train_input1, train_input2, train_label in train_loader:
                model.train()
                train_input1 = train_input1.to(device=device, dtype=torch.float32)
                train_input2 = train_input2.to(device=device, dtype=torch.float32)
                train_label = train_label.to(device=device, dtype=torch.float32)
                embedding1, embedding2 = model.forward(
                    train_input1, train_input2)
                # decoder1, decoder2 = model.decoder(embedding1, embedding2)
                optimizer.zero_grad()
                supervised_loss = loss_function(embedding1.double(), embedding2.double(), train_label.double())
                supervised_loss = supervised_loss.to(device)
                supervised_loss.backward()
                optimizer.step()

                current_loss += supervised_loss.item()
            current_loss /= len(train_loader)
            loss_list.append(current_loss)

        # scheduler.step()

    logger.info('Training finished.')
    torch.save(model, out)

    with open(out + '.loss', 'w') as f:
        for idx, loss in enumerate(loss_list):
            f.write(f'{idx}\t{loss}\n')

    return model
