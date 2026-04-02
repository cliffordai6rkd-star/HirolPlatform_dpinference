from __future__ import annotations
from typing import List
from bc_utils.utils import LossType
from models.encoder import ResNetEncoder, MLP

import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def load_model(model_path, device, training=True):
    ckpt = th.load(model_path, map_location=device)
    args = ckpt["args"]
    rgb_keys = ckpt["rgb_keys"]
    proprio_dim = ckpt["proprio_dim"]
    action_dim = ckpt["action_dim"]
    loss_type = LossType.MSE if args.get("loss", "mse") == "mse" else LossType.LOG_PROB
    
    model = MlpBcPolicy(
        rgb_keys=rgb_keys,
        share_rgb=args.get("share_rgb", True),
        rgb_encoder_name=args.get("rgb_encoder", "resnet18"),
        pretrained_backbone=not args.get("no_pretrained", False),
        freeze_resnet=args.get("freeze_backbone", False),
        proprio_dim=proprio_dim,
        proprio_feature_size=args.get("proprio_feature", 32),
        action_dim=action_dim,
        hidden_dims=args.get("hidden_dims", [512, 512, 512]),
        loss_type=loss_type,
        training=training
    ).to(device)
    
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print("[load warning] missing:", missing, "unexpected:", unexpected)
    
    return model, args

class MlpBcPolicy(nn.Module):
    def __init__(self, rgb_keys, share_rgb=False, rgb_encoder_name="resnet18", freeze_resnet=False,
                 proprio_dim=8, proprio_feature_size=5, action_dim=7, hidden_dims=[512,512,512],
                 pretrained_backbone = True, loss_type=LossType.MSE, dropout=0.25, 
                 use_layer_norm=True, training=True):
        super().__init__()
        self.loss_type = loss_type
        self.share_rgb = share_rgb
        self.rgb_keys = rgb_keys
        self.training = training
        
        self.num_rgb_features = 0
        if self.share_rgb:
            self.rgb_encoders = ResNetEncoder(name=rgb_encoder_name, 
                trainable=not freeze_resnet, pretrained=pretrained_backbone)
            self.num_rgb_features = self.rgb_encoders.out_dim * len(self.rgb_keys)
        else:
            self.rgb_encoders = nn.ModuleDict({
                k: ResNetEncoder(
                    name=rgb_encoder_name, trainable=not freeze_resnet,
                    pretrained=pretrained_backbone) for k in self.rgb_keys})
            self.num_rgb_features = sum(self.rgb_encoders[k].out_dim for k in self.rgb_keys)
        
        pro_layers: List[nn.Module] = [nn.Linear(proprio_dim, proprio_feature_size)]
        if use_layer_norm:
            pro_layers.append(nn.LayerNorm(proprio_feature_size))
        pro_layers.append(nn.SiLU())
        self.proprio_net = nn.Sequential(*pro_layers)
        dropout = 0.0 if not training else dropout
        self.mlp = MLP(input_dim=self.num_rgb_features+proprio_feature_size, 
                       output_dim=hidden_dims[-1], hidden_dims=hidden_dims,
                       dropout_rate=dropout, use_layer_norm=use_layer_norm)

        if self.loss_type == LossType.MSE:
            self.action_fc = nn.Linear(in_features=hidden_dims[-1], out_features= action_dim)
        elif self.loss_type == LossType.LOG_PROB:
            self.mu_fc = nn.Linear(in_features=hidden_dims[-1], out_features=action_dim)
            self.logvar_fc = nn.Linear(in_features=hidden_dims[-1], out_features=action_dim)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def forward(self, imgs, state):
        B = state.shape[0]
        rgb_features = None
        if self.share_rgb:
            cocat_imgs = None
            for key in self.rgb_keys:
                if key not in imgs:
                    raise ValueError(f'Input images do not have the required key: {key}')
                # B C H W -> (B*N C H W)
                cocat_imgs = imgs[key] if cocat_imgs is None else th.cat((cocat_imgs, imgs[key]), dim=0)
            rgb_features:Tensor = self.rgb_encoders(cocat_imgs)
            # B*N D -> B D*N
            rgb_features = rgb_features.view(B, -1)
        else:
            for key in self.rgb_keys:
                if key not in imgs:
                    raise ValueError(f'Input images do not have the required key: {key}')
                img_feature = self.rgb_encoders[key](imgs[key])
                # B D -> (B D*N)
                rgb_features = img_feature if rgb_features is None else th.cat((rgb_features, img_feature), dim=1)
        
        state_feature = self.proprio_net(state)
        obs_feature = th.cat((rgb_features, state_feature), dim=1)
        obs_feature = self.mlp(obs_feature)
        
        if self.loss_type == LossType.MSE:
            pred_action = self.action_fc(obs_feature)
            return pred_action
            
        elif self.loss_type == LossType.LOG_PROB:
            action_mu = self.mu_fc(obs_feature)
            action_logvar = self.logvar_fc(obs_feature)
            pred_action = self.sample_action(action_mu, action_logvar)
            return pred_action, action_mu, action_logvar
        else:
            raise NotImplementedError
    
    def compute_mse_loss(self, gt_x, pred_x):
        if self.loss_type != LossType.MSE:
            raise ValueError("compute_mse_loss can only be used with MSE loss type.")
        else: return F.mse_loss(pred_x, gt_x)
    
    def compute_log_prob_loss(self, gt_x, mu, log_var):
        if self.loss_type != LossType.LOG_PROB:
            raise ValueError("compute_log_prob_loss can only be used with LOG_PROB loss type.")
        elif self.loss_type == LossType.LOG_PROB:
            # loss = F.gaussian_nll_loss
            scale = th.exp(0.5 * log_var)
            dist = th.distributions.Independent(
                th.distributions.Normal(loc=mu, scale=scale), 1
            )
            loss = -dist.log_prob(gt_x).mean()
            return loss
        else: 
            raise NotImplementedError
                    
    def sample_action(self, mean, log_var):
        scale = th.exp(0.5 * log_var)
        dist = th.distributions.Independent(
            th.distributions.Normal(loc=mean, scale=scale), 1)
        return dist.sample() if self.training else dist.mean()
    