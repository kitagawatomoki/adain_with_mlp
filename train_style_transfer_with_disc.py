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

from utils import (return_img, make_grid,img_collate_func,
                    CLIP_fn, LPIPS, hinge_d_loss, vanilla_d_loss,
                    calc_mean_std, calculate_adaptive_weight, adopt_weight)
from model import Discriminator

def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-f", "--save_folder", type=str, default="result")
    parser.add_argument("-mm", "--model_mood", type=str, default="with_clip") # 使用するモデルの種類
    parser.add_argument("-dm", "--data_mood", type=str, default="font_indices") # 使用するデータローダの種類
    parser.add_argument("--step_type", type=str, default="step1") # 使用するデータローダの訓練ステップ
    parser.add_argument("-s", "--img_size", type=int, default=256) # 画像サイズ(256しか試していない，128でも学習はできると思う)
    parser.add_argument("-b", "--batch_size", type=int, default=8) # バッチサイズ(大きくするとGPUに乗り切らない，4,8を推奨)
    parser.add_argument("-lr", "--learing_late", type=float, default=4.5e-6) # 学習率(Discriminatorがある場合は固定しないと学習できない恐れがある)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--use_char", type=str, default="information/hira_kana_kanji.txt")
    parser.add_argument("--use_root", type=str, default="information/char_paths/train")
    parser.add_argument("--save_num", type=str, default=None) # 使用する手書き画像の枚数の設定(少ない訓練データでも学習できるかの実験のため)
    parser.add_argument("--gen_weight", type=str, default=None) # 途中から生成モデルを学習する際に設定する
    parser.add_argument("--disc_weight", type=str, default=None) # 途中から判別モデルを学習する際に設定する
    args = parser.parse_args()
    config = vars(args)

    device = torch.device('cuda:%d'%(config["gpu"]) if torch.cuda.is_available() else 'cpu')

    if config["save_num"] is not None:
        save_num = "_hand-{}".format(config["save_num"])
    else:
        save_num = ""
    save_folder = "{}_with_disc_model-{}_data-{}_step-{}{}".format(config["save_folder"],
                                                                config["model_mood"],
                                                                config["data_mood"],
                                                                config["step_type"],
                                                                save_num)
    save_log = os.path.join(save_folder, 'log')
    save_model = os.path.join(save_folder, 'model')
    save_image = os.path.join(save_folder, 'image')
    os.makedirs(save_log, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)
    os.makedirs(save_image, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=save_log)
    with open(os.path.join(save_folder, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    if config["model_mood"] == "with_clip":
        from model_with_clip import AdaIN_with_CLIP
        collate_fn = img_collate_func

        model = AdaIN_with_CLIP(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            cond_drop_prob = 0.5
        ).to(device)

    elif config["model_mood"] == "with_mlp":
        from model_with_mlp import AdaIN_with_MLP
        with open(config["use_char"]) as f:
            char_list = [s.strip() for s in f.readlines()]
        collate_fn = None

        model = AdaIN_with_MLP(
            dim = 64,
            num_classes = len(char_list),
            dim_mults = (1, 2, 4, 8),
            cond_drop_prob = 0.5
        ).to(device)

    else:
        from model_without_mlp import AdaIN_without_MLP
        collate_fn = None

        model = AdaIN_without_MLP(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            cond_drop_prob = 0.5
        ).to(device)

    discriminator = Discriminator()
    discriminator = discriminator.to(device)

    if config["gen_weight"] is not None:
        gen_weight = torch.load(config["gen_weight"], map_location=device)
        model.load_state_dict(gen_weight)

    if config["disc_weight"] is not None:
        disc_weight = torch.load(config["disc_weight"], map_location=device)
        discriminator.load_state_dict(disc_weight)


    if config["data_mood"] == "font_indices":
        from dataset_font_indices import Dataset
    else:
        from dataset_char_indices import Dataset

    train_dataset = Dataset(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=collate_fn)


    gen_opt = torch.optim.Adam(model.parameters(), lr=config["learing_late"], betas=(0.5, 0.9))
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=config["learing_late"], betas=(0.5, 0.9))

    clip_loss_fn = CLIP_fn()
    reco_loss_fn = nn.MSELoss()
    implicit_loss_fn = nn.MSELoss()
    implicit_weight = 10

    perceptual_weight = 1.0
    perceptual_loss = LPIPS(device).eval()
    cond=None
    disc_conditional = False
    discriminator_weight = 1.0
    disc_factor_weight = 1.0
    discriminator_iter_start = 0
    disc_loss="hinge"
    if disc_loss == "hinge":
        disc_loss = hinge_d_loss
    elif disc_loss == "vanilla":
        disc_loss = vanilla_d_loss

    train_step = 0
    epochs = config["epochs"]
    for epoch in range(epochs):
        model.train()#学習モードに移行
        discriminator.train()
        training=True

        train_loss = 0
        s_time = time.time()
        for batch in tqdm(train_dataloader):
            image, s_image, l_label, l_tokens_tensor,l_token_type_ids_tensor,l_attention_mask_tensor = batch
            image, s_image, l_label, l_tokens_tensor,l_token_type_ids_tensor,l_attention_mask_tensor = image.to(device), s_image.to(device), l_label.to(device), l_tokens_tensor.to(device),l_token_type_ids_tensor.to(device),l_attention_mask_tensor.to(device)
            l_inputs = {"input_ids":l_tokens_tensor, "token_type_ids":l_token_type_ids_tensor, "attention_mask":l_attention_mask_tensor}

            if config["model_mood"] == "with_clip":
                quant, s_quant, c_quant = model.encode(image, s_image, l_inputs)
                dec = model.decode(quant, c_quant)
                l_s_quant = model.style_encoder(dec)
            elif config["model_mood"] == "with_mlp":
                quant, s_quant, c_quant = model.encode(image, s_image, l_label)
                dec = model.decode(quant, c_quant)
                l_s_quant = model.style_encoder(dec)
            else:
                quant, s_quant = model.encode(image, s_image)
                dec = model.decode(quant)
                l_s_quant = model.style_encoder(dec)

            gen_opt.zero_grad()

            explicit_loss = clip_loss_fn.compute_losses(dec, l_inputs).mean()

            s_feat_mean, s_feat_std = calc_mean_std(s_quant)
            l_s_feat_mean, l_s_feat_std = calc_mean_std(l_s_quant)
            implicit_loss = implicit_loss_fn(s_feat_mean, l_s_feat_mean) + implicit_loss_fn(s_feat_std, l_s_feat_std)

            if cond is None:
                assert not disc_conditional
                logits_fake = discriminator(dec.contiguous())
            else:
                assert disc_conditional
                logits_fake = discriminator(torch.cat((dec.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            rec_loss = torch.abs(s_image.contiguous() - dec.contiguous())
            if perceptual_weight > 0:
                p_loss = perceptual_loss(s_image.contiguous(), dec.contiguous())
                rec_loss = rec_loss + perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])

            nll_loss = rec_loss
            nll_loss = torch.mean(nll_loss)
            try:
                d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer=model.final_conv.weight)
            except RuntimeError:
                assert not training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(disc_factor_weight, train_step, threshold=discriminator_iter_start)

            total_g_step_loss = explicit_loss + implicit_weight*implicit_loss + nll_loss + d_weight * disc_factor * g_loss
            train_loss+=float(total_g_step_loss.to('cpu').detach().numpy().copy())

            if train_step % 1000 == 0:
                org_images = return_img(image)
                style_images = return_img(s_image)
                mix_images = return_img(dec)

                save_img = np.concatenate([org_images, style_images, mix_images], 0)
                save_img = make_grid(save_img, args.batch_size)
                cv2.imwrite(os.path.join(save_image, "train_ep{}-step{}.png".format(epoch, train_step)), save_img)

            summary_writer.add_scalar("train/total_loss", total_g_step_loss.clone().detach(), train_step)
            summary_writer.add_scalar("train/nll_loss", nll_loss.clone().detach().mean(), train_step)
            summary_writer.add_scalar("train/d_weight", d_weight.clone().detach().mean(), train_step)
            summary_writer.add_scalar("train/disc_factor", disc_factor, train_step)
            summary_writer.add_scalar("train/g_loss", g_loss.detach().mean(), train_step)
            summary_writer.add_scalar("train/explicit_loss", explicit_loss.clone().detach(), train_step)
            summary_writer.add_scalar("train/implicit_loss", implicit_loss.clone().detach(), train_step)

            total_g_step_loss.backward()
            gen_opt.step()

            # discriminator update
            dis_opt.zero_grad()

            if cond is None:
                logits_real = discriminator(s_image.contiguous().detach())
                logits_fake = discriminator(dec.contiguous().detach())
            else:
                logits_real = discriminator(torch.cat((s_image.contiguous().detach(), cond), dim=1))
                logits_fake = discriminator(torch.cat((dec.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(disc_factor_weight, train_step, threshold=discriminator_iter_start)
            total_d_step_loss = disc_factor * disc_loss(logits_real, logits_fake)

            summary_writer.add_scalar("train/disc_loss", total_d_step_loss.clone().detach().mean(), train_step)
            summary_writer.add_scalar("train/logits_real", logits_real.clone().detach().mean(), train_step)
            summary_writer.add_scalar("train/logits_fake", logits_fake.clone().detach().mean(), train_step)

            total_d_step_loss.backward()
            dis_opt.step()

            if train_step % 25000 == 0:
                torch.save(model.state_dict(), os.path.join(save_model, "u-net_ae_model_step%s.pt"%(train_step)))
                torch.save(discriminator.state_dict(), os.path.join(save_model, "discriminator_ep_step%s.pt"%(train_step)))

            train_step+=1
        train_loss/=len(train_dataloader)

        print('Time for epoch {} is {} | Train Loss {}'.format(epoch+1, time.time() - s_time, train_loss))

    torch.save(model.state_dict(), os.path.join(save_model, "u-net_ae_model.pt"))
    torch.save(discriminator.state_dict(), os.path.join(save_model, "discriminator.pt"))