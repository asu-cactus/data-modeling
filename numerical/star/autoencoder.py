import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb
import logging

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s", filemode="w")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


def load_data(column):
    # normalize_factors = {
    #     0: 1e-3,  # charge
    #     1: 1e-4,  # clus
    #     2: 1e-6,  # dst
    #     3: 1e-6,  # hist
    #     4: 1e-5,  # enumber
    #     5: 1e-8,  # etime
    #     6: 1e-7,  # rnumber
    #     7: 1e-4,  # nlb
    #     8: 1e-2,  # qxb
    #     9: 1e-4,  # tracks
    #     10: 1e-3,  # vertex
    #     11: 1e-3,  # zdc
    # }

    data = pd.read_csv(
        "data/star2000.csv.gz",
        header=None,
        # nrows=NROWS,
        nrows=2_000_000,
        usecols=[column],
        dtype=np.float32,  # Make data type to be float32
    )
    data = data.to_numpy().reshape((-1, NROWS))
    # data = data * normalize_factors[column]  # Normalize data
    return data


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(NROWS, 6000),
            nn.ReLU(),
            nn.Linear(6000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 50),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(50, 1000),
            nn.ReLU(),
            nn.Linear(1000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 6000),
            nn.ReLU(),
            nn.Linear(6000, NROWS),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


def run(data):
    data = torch.from_numpy(data)
    criterion = nn.MSELoss()

    encoder = Encoder().to("cuda:0")
    decoder = Decoder().to("cuda:1")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=500, verbose=True)

    smallest_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        # ===================forward=====================
        output = encoder(data.to("cuda:0"))
        output = decoder(output.to("cuda:1"))
        loss = criterion(output, data.to("cuda:1"))
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        # ===================log========================
        smallest_loss = min(smallest_loss, loss.item())
        if epoch % 100 == 0:
            logger.info(
                f"epoch [{epoch}/{num_epochs}], loss:{loss.item()}, smallest_loss: {smallest_loss}"
            )
    # torch.save(encoder.state_dict(), "./outputs/encoder.pth")
    # torch.save(decoder.state_dict(), "./outputs/decoder.pth")


if __name__ == "__main__":
    # names = {
    #     "charge": np.int32,
    #     "clus": np.int32,
    #     "dst": np.int32,
    #     "hist": np.int32,
    #     "enumber": np.int32,
    #     "etime": np.float32,
    #     "rnumber": np.int32,
    #     "nlb": np.int32,
    #     "qxb": np.float32,
    #     "tracks": np.int32,
    #     "vertex": np.float32,
    #     "zdc": np.int32,
    # }

    NROWS = 20_000  # total rows = 2_173_762
    num_epochs = 50_000
    learning_rate = 1e-5

    column = 2
    data = load_data(column)
    run(data)
