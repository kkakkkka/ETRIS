import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

from .layers import HA, GA, Projector
from .bridger import Bridger_RN as Bridger_RL, Bridger_ViT as Bridger_VL


class ETRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        if "RN" in cfg.clip_pretrain:
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
            self.bridger = Bridger_RL(d_model=cfg.ladder_dim, nhead=cfg.nhead, fusion_stage=cfg.multi_stage)
        else:
            self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size).float()
            self.bridger = Bridger_VL(d_model=cfg.ladder_dim, nhead=cfg.nhead)
        
        # Fix Backbone
        for param_name, param in self.backbone.named_parameters():
            if 'positional_embedding' not in param_name:
                param.requires_grad = False       

        # Multi-Modal Decoder
        self.neck = HA(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        self.decoder = GA(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)

        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis, word, state = self.bridger(img, word, self.backbone)

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            return pred.detach(), mask, loss
        else:
            return pred.detach()
