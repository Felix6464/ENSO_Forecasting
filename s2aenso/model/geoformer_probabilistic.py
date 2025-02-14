import torch
import torch.nn as nn
from s2aenso.utils.geoformer_utils import *


class Geoformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_size = config["d_size"]
        self.device = config["device"]
        if self.config["needtauxy"]:
            self.cube_dim = (
                (config["input_channel"] + 2) * config["patch_size"][0] * config["patch_size"][1]
            )
        else:
            self.cube_dim = (
                config["input_channel"] * config["patch_size"][0] * config["patch_size"][1]
            )
        self.predictor_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=d_size,
            emb_spatial_size=config["emb_spatial_size"],
            max_len=config["input_length"],
            device=self.device,
        )
        self.predictand_emb_train = make_embedding(
            cube_dim=self.cube_dim,
            d_size=d_size,
            emb_spatial_size=config["emb_spatial_size"],
            max_len=config["output_length"],
            device=self.device,
        )
        #self.predictand_emb_val = make_embedding(
        #    cube_dim=self.cube_dim,
        #    d_size=d_size,
        #    emb_spatial_size=config["emb_spatial_size"],
        #    max_len=config["output_length_val"],
        #    device=self.device,
        #)
        enc_layer = miniEncoder(
            d_size, config["nheads"], config["dim_feedforward"], config["dropout"]
        )
        dec_layer = miniDecoder(
            d_size, config["nheads"], config["dim_feedforward"], config["dropout"]
        )
        self.encoder = multi_enc_layer(
            enc_layer=enc_layer, num_layers=config["num_encoder_layers"]
        )
        self.decoder = multi_dec_layer(
            dec_layer=dec_layer, num_layers=config["num_decoder_layers"]
        )
        self.linear_output = nn.Linear(d_size, self.cube_dim)
        self.linear_output2 = nn.Linear(d_size, self.cube_dim)

    def forward(
        self,
        predictor,
        predictand,
        config,
        in_mask=None,
        enout_mask=None,
        train=True,
        sv_ratio=0,
    ):
        """
        Args:
            predictor: (batch, lb, C, H, W)
            predictand: (batch, pre_len, C, H, W)
        Returns:
            outvar_pred: (batch, pre_len, C, H, W)
        """

        predictor = predictor.permute(0, 2, 1, 3, 4)
        predictand = predictand.permute(0, 2, 1, 3, 4)


        en_out = self.encode(predictor=predictor, in_mask=in_mask)
        #print("Encoder output shape: ", en_out.shape)
        if train:
            with torch.no_grad():
                #print("predictor[:, -1:]", predictor[:, -1:].shape)
                #print("predictand[:, :-1]", predictand[:, :-1].shape)
                connect_inout = torch.cat(
                    [predictor[:, -1:], predictand[:, :-1]], dim=1
                )
                out_mask = self.make_mask_matrix(connect_inout.size(1))
                outvar_pred = self.decode(
                    connect_inout,
                    en_out,
                    out_mask,
                    enout_mask,
                    mode="train"
                )
            #print("Outvar_pred shape:", outvar_pred.shape)
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio
                    * torch.ones(predictand.size(0), predictand.size(1) - 1, 1, 1, 1)
                ).to(self.device)
            else:
                supervise_mask = 0
            predictand = (
                supervise_mask * predictand[:, :-1]
                + (1 - supervise_mask) * outvar_pred[:, 0, :-1]
            )
            predictand = torch.cat([predictor[:, -1:], predictand], dim=1)
            # predicting
            outvar_pred = self.decode(
                predictand,
                en_out,
                out_mask,
                enout_mask,
                mode="train"
            )
        else:
            predictand = None
            assert predictand is None
            predictand = predictor[:, -1:]
            #print("Predictand shape: ", predictand.shape)
            for t in range(self.config["output_length"]):
                out_mask = self.make_mask_matrix(predictand.size(1))
                outvar_pred = self.decode(
                    predictand,
                    en_out,
                    out_mask,
                    enout_mask,
                    mode="val"
                )
                predictand = torch.cat([predictand, outvar_pred[:, 0, -1:]], dim=1)
                #print("Outvar_pred shape:", outvar_pred.shape)
                #print("Predictand shape:", predictand.shape)

        return outvar_pred

    def encode(self, predictor, in_mask):
        """
        predictor: (B, lb, C, H, W)
        en_out: (Batch, S, lb, d_size)
        """
        
        lb = predictor.size(1)
        #("Sequence length: ", lb)
        #print("Predictor shape permute to match predictor: (B, lb, C, H, W)")
        #print("Predictor shape before unfold: ", predictor.shape)
        predictor = unfold_func(predictor, self.config["patch_size"])
        #print("Predictor shape after unfold:", predictor.shape)
        #print("Cube dim: ", self.cube_dim)
        predictor = predictor.reshape(predictor.size(0), lb, self.cube_dim, -1).permute(
            0, 3, 1, 2
        )
        #print("Predictor shape before embedding: ", predictor.shape)
        predictor = self.predictor_emb(predictor)
        en_out = self.encoder(predictor, in_mask)
        return en_out

    def decode(self, predictand, en_out, out_mask, enout_mask, mode):
        """
        Args:
            predictand: (B, pre_len, C, H, W)
        output:
            (B, pre_len, C, H, W)
        """
        H, W = predictand.size()[-2:]
        #print("H: ", H)
        #print("W: ", W)
        T = predictand.size(1)
        #print("T: ", T)
        #print("DECODE")
        #print("Predictand shape before unfold: ", predictand.shape)
        predictand = unfold_func(predictand, self.config["patch_size"])
        #print("Predictand shape after unfold: ", predictand.shape)
        predictand = predictand.reshape(
            predictand.size(0), T, self.cube_dim, -1
        ).permute(0, 3, 1, 2)
        #print("Predictand shape before embedding: ", predictand.shape)
        #if mode == "train":
        #    predictand = self.predictand_emb_train(predictand)
        #else:
        #    predictand = self.predictand_emb_val(predictand)

        predictand = self.predictand_emb_train(predictand)

        #print("Predictand shape after embedding: ", predictand.shape)
        output_ = self.decoder(predictand, en_out, out_mask, enout_mask)
        output = self.linear_output(output_).permute(0, 2, 3, 1)
        output2 = self.linear_output2(output_).permute(0, 2, 3, 1)
        output = output.reshape(
            predictand.size(0),
            T,
            self.cube_dim,
            H // self.config["patch_size"][0],
            W // self.config["patch_size"][1],
        )
        output = fold_func(
            output, output_size=(H, W), kernel_size=self.config["patch_size"]
        )
        output2 = output2.reshape(
            predictand.size(0),
            T,
            self.cube_dim,
            H // self.config["patch_size"][0],
            W // self.config["patch_size"][1],
        )
        output2 = fold_func(
            output2, output_size=(H, W), kernel_size=self.config["patch_size"]
        )
        return torch.stack([output, output2], dim=1)

    def make_mask_matrix(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.config["device"])


class multi_enc_layer(nn.Module):
    def __init__(self, enc_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([enc_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class multi_dec_layer(nn.Module):
    def __init__(self, dec_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([dec_layer for _ in range(num_layers)])

    def forward(self, x, en_out, out_mask, enout_mask):
        for layer in self.layers:
            x = layer(x, en_out, out_mask, enout_mask)
        return x
