import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
import numpy as np
import os
import json
from tqdm import tqdm
import time
import cv2

from dataset import Dataset
from model import AutoEncoder
from utils import return_img, make_grid

def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-f", "--save_folder", type=str, default="result")
    parser.add_argument("-s", "--img_size", type=int, default=256)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-lr", "--learing_late", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--use_char", type=str, default="information/hira_kana_kanji.txt")
    parser.add_argument("--use_root", type=str, default="information/char_paths/train")
    parser.add_argument("--save_num", type=str, default=None)
    args = parser.parse_args()
    config = vars(args)

    device = torch.device('cuda:%d'%(config["gpu"]) if torch.cuda.is_available() else 'cpu')

    save_folder = "experiment/{}_reco".format(config["save_folder"])
    save_log = os.path.join(save_folder, 'log')
    save_model = os.path.join(save_folder, 'model')
    save_image = os.path.join(save_folder, 'image')
    os.makedirs(save_log, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)
    os.makedirs(save_image, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=save_log)
    with open(os.path.join(save_folder, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    train_dataset = Dataset(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)

    model = AutoEncoder(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learing_late"])
    reco_loss_fn = nn.MSELoss()

    train_step = 0
    epochs = config["epochs"]
    for epoch in range(epochs):
        model.train()#学習モードに移行
        train_loss = 0
        s_time = time.time()
        for batch1 in tqdm(train_dataloader):
            image = batch1
            image = image.to(device)

            optimizer.zero_grad()

            reco = model(image)

            loss = reco_loss_fn(image, reco)
            train_loss+=float(loss.to('cpu').detach().numpy().copy())

            if train_step % 1000 == 0:
                org_images = return_img(image)
                reco_images = return_img(reco)

                save_img = np.concatenate([org_images, reco_images], 0)
                save_img = make_grid(save_img, config["batch_size"])
                cv2.imwrite(os.path.join(save_image, "train_ep{}-step{}.png".format(epoch, train_step)), save_img)

            summary_writer.add_scalar("loss", loss.clone().detach(), train_step)

            loss.backward()
            optimizer.step()

            train_step+=1

        print('Time for epoch {} is {} | Train Loss {}'.format(epoch, time.time() - s_time, train_loss))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_model, "ae_model_ep%s.pt"%(epoch)))
    torch.save(model.state_dict(), os.path.join(save_model, "AutoEncoder.pt"))


if __name__ == "__main__":
    main()