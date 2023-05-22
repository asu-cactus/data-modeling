import pandas as pd
import numpy as np
from torch import nn
import torch
from torcheval.metrics import MulticlassAccuracy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import deque
import os
import logging

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_model(out_units, checkpoint=None):
    model = nn.Sequential(
        nn.Linear(1, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, out_units),
    )
    if checkpoint is not None:
        model.load_state_dict(torch.load(f"outputs/{checkpoint}"))
    return model


def load_data(usecols, name, dtype):
    data = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        usecols=[usecols],
        names=[name],
        dtype=dtype,
    )
    return data


def to_categorical(data, name):
    unique_values = data[name].unique()
    unique_values = sorted(unique_values)
    mapping = {value: i for i, value in enumerate(unique_values)}
    # Transform data with mapping
    data[name] = data[name].map(mapping)
    return len(unique_values)


def create_training_loader(data, name, batch_size=256):
    # Create a TensorDataset from our data
    dataset = torch.utils.data.TensorDataset(
        torch.arange(len(data), dtype=torch.float32).unsqueeze(1),
        torch.tensor(data[name].values, dtype=torch.long),
    )
    # Create a DataLoader from the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_one_epoch(
    model, optimizer, scheduler, loss_fn, metric, epoch_index, tb_writer
):
    running_loss = 0.0
    last_loss = 0.0

    for _, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Calculate accuracy
        metric.update(
            torch.argmax(outputs, dim=1).detach().cpu(), labels.detach().cpu()
        )

    acc = metric.compute().item()
    metric.reset()

    avg_loss = running_loss / len(training_loader)  # loss per batch
    scheduler.step(avg_loss)
    logging.info(f"Epoch {epoch_index} loss: {avg_loss}, accuracy: {acc}")
    tb_writer.add_scalar("Loss/train", avg_loss, epoch_index)
    tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_index)
    tb_writer.add_scalar("acc", acc, epoch_index)
    tb_writer.flush()
    return last_loss


def train(model, optimizer, scheduler, loss_fn, metric, epochs=10000):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("outputs/runs/mlp_{}".format(timestamp))
    checkpoints = deque()
    maxlen = 5

    model.train(True)
    model = model.to(device)
    for epoch_number in range(epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        # model.train(True)
        train_one_epoch(
            model, optimizer, scheduler, loss_fn, metric, epoch_number, writer
        )

        model_path = "outputs/model_{}_{}".format(timestamp, epoch_number)
        checkpoints.append(model_path)
        if len(checkpoints) > maxlen:
            old_model_path = checkpoints.popleft()
            os.remove(old_model_path)
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    usecols = 1
    name = "clus"
    dtype = np.int32
    data = load_data(usecols, name, dtype)
    num_cate = to_categorical(data, name)
    training_loader = create_training_loader(data, name)

    model = get_model(num_cate)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    metric = MulticlassAccuracy(num_classes=num_cate)
    train(model, optimizer, scheduler, loss_fn, metric)
