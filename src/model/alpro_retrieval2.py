"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import BertTokenizer
from transformers import WhisperModel

from src.model.alpro_outputs import AlproIntermediateOutput, AlproOutput
from src.model.med import XBertEncoder
from src.model.timesformer.vit import TimeSformer
from src.model.lstm_compression import LSTM_fc
from src.model.base_model import BaseModel

class MAACAPretrain(BaseModel):

    def __init__(
        self, visual_encoder, text_encoder, text_encoder2, audio_encoder, config, temp=0.7
    ):
        super().__init__()

        self.model_config = config
        self.temp = nn.Parameter(torch.ones([]) * temp)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.text_encoder2 = text_encoder2
        self.audio_encoder = audio_encoder

        self.transform_audio_to_hidden = LSTM_fc(input_size=768,
                                                 hidden_size=self.model_config.audio_transform_hidden_dim,
                                                 num_layers=self.model_config.audio_transform_num_layers,
                                                 output_seq_len=self.model_config.audio_output_seq_len,
                                                 output_size=self.model_config.audio_transform_output_dim)
        self.transform_video_to_hidden = LSTM_fc(input_size=768,
                                                 hidden_size=self.model_config.video_transform_hidden_dim,
                                                 num_layers=self.model_config.video_transform_num_layers,
                                                 output_seq_len=self.model_config.video_output_seq_len,
                                                 output_size=self.model_config.video_transform_output_dim)

        self.max_txt_len = self.model_config.max_txt_len

        self.itm_head = nn.Linear(768, 2)
        self.atm_head = nn.Linear(768, 2)

        self.max_txt_len = self.model_config.max_txt_len
    
    def forward(self, video, audio, text_input, aspect, complaint, is_train=True):
        visual_inputs = video
        audio_inputs = audio
        caption = text_input

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        b, c, t, h, w = visual_inputs.shape

        # forward text
        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
      
        # for text-video
        text_output = self.text_encoder.forward_text(
            text,
            token_type_ids=torch.zeros(
                text.input_ids.shape, dtype=torch.long, device=self.device
            ),
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(text_embeds[:, 0, :], dim=-1)

        # for text-audio
        text_output2 = self.text_encoder2.forward_text(
            text,
            token_type_ids=torch.zeros(
                text.input_ids.shape, dtype=torch.long, device=self.device
            ),
        )
        text_embeds2 = text_output2.last_hidden_state
        text_feat2 = F.normalize(text_embeds2[:, 0, :], dim=-1)

        # forward visual
        # timeSformer asks for (b, c, t, h, w) as input.
        video_embeds = self.visual_encoder.forward_features(visual_inputs)
        # 
        # import pdb; pdb.set_trace()
        video_embeds = self.transform_video_to_hidden(video_embeds)

        video_feat = F.normalize(video_embeds[:, 0, :], dim=-1)

        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        audio_embeds = self.audio_encoder(audio_inputs).last_hidden_state
        audio_embeds = self.transform_audio_to_hidden(audio_embeds)
        audio_feat = F.normalize(audio_embeds[:, 0, :], dim=-1)

        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        # ========== (in-batch) ITC loss ==========

        gathered_video_feats = video_feat
        gathered_text_feats = text_feat
        gathered_text_feats2 = text_feat2
        gathered_audio_feats = audio_feat

        sim_v2t = video_feat @ gathered_text_feats.t() / self.temp
        sim_t2v = text_feat @ gathered_video_feats.t() / self.temp

        sim_a2t = audio_feat @ gathered_text_feats2.t() / self.temp
        sim_t2a = text_feat @ gathered_audio_feats.t() / self.temp

        sim_targets = torch.zeros_like(sim_v2t)

        local_rank = 0
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start:b_end] = torch.eye(b)

        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * sim_targets, dim=1).mean()

        loss_a2t = -torch.sum(F.log_softmax(sim_a2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2a = -torch.sum(F.log_softmax(sim_t2a, dim=1) * sim_targets, dim=1).mean()

        vtc_loss = (loss_v2t + loss_t2v) / 2
        atc_loss = (loss_a2t + loss_t2a) / 2

        (
            vtm_loss,
            vtm_logits,
            vtm_labels,
            encoder_output,
            encoder_output_neg
        ) = self.compute_vtm(
            text_embeds=text_embeds,
            text_atts=text.attention_mask,
            image_embeds=video_embeds,
            image_atts=video_atts,
            sim_i2t=sim_v2t.clone(),  # for hard mining
            sim_t2i=sim_t2v.clone(),  # for hard mining
            modality="video"
        )

        
        (
            atm_loss,
            atm_logits,
            atm_labels,
            encoder_output2,
            encoder_output_neg2
        ) = self.compute_vtm(
            text_embeds=text_embeds2,
            text_atts=text.attention_mask,
            image_embeds=audio_embeds,
            image_atts=audio_atts,
            sim_i2t=sim_a2t.clone(),  # for hard mining
            sim_t2i=sim_t2a.clone(),  # for hard mining
            modality="audio"
        )


        loss = vtc_loss + atc_loss + vtm_loss + atm_loss

        return AlproOutput(
            loss=loss,
            loss_vtc=vtc_loss,
            loss_vtm=vtm_loss,
            intermediate_output=AlproIntermediateOutput(
                video_embeds=video_embeds,
                text_embeds=text_embeds,
                encoder_output=encoder_output,
                encoder_output_neg=encoder_output_neg,
                vtm_logits=vtm_logits,
                vtm_labels=vtm_labels,
            ),
        )

    def compute_vtm(
        self, text_embeds, text_atts, image_embeds, image_atts, sim_i2t, sim_t2i, modality
    ):
        device = self.device
        text_encoder = self.text_encoder if modality == "video" else self.text_encoder2

        # ====== positive pairs =======
        attention_mask = torch.cat([text_atts, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs_pos = text_encoder(
            encoder_embeds=embedding_output_pos,
            attention_mask=attention_mask,
            return_dict=True,
            mode="fusion",
        )

        # ====== negative pairs =======
        bs = text_embeds.shape[0]

        # local_rank = get_rank()
        local_rank = 0
        b_start, b_end = bs * local_rank, bs * (local_rank + 1)

        with torch.no_grad():
            weights_v2t = sim_i2t[:, b_start:b_end]
            weights_t2v = sim_t2i[:, b_start:b_end]

            # never select self as negative
            weights_v2t.fill_diagonal_(-np.Inf)
            weights_t2v.fill_diagonal_(-np.Inf)

            weights_v2t = F.softmax(weights_v2t, dim=1)
            weights_t2v = F.softmax(weights_t2v, dim=1)

        # select a negative image for each text
        # FIXME to optimize using indexing operations
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2v[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_v2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)

        video_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        video_atts_all = torch.cat([image_atts, image_atts], dim=0)

        attention_mask_all = torch.cat([text_atts_all, video_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, video_embeds_all], dim=1) # b, 128 + 197, 768

        # forward negative pairs via cross encoder
        encoder_outputs_neg = text_encoder(
            encoder_embeds=embedding_output_all,
            attention_mask=attention_mask_all,
            return_dict=True,
            mode="fusion",
        )

        vl_embeddings = torch.cat(
            [
                encoder_outputs_pos.last_hidden_state[:, 0, :],
                encoder_outputs_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )

        if modality == "audio":
            vtm_logits = self.atm_head(vl_embeddings)
        else:
            vtm_logits = self.itm_head(vl_embeddings)

        vtm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)

        return (
            vtm_loss,
            vtm_logits,
            vtm_labels,
            encoder_outputs_pos,
            encoder_outputs_neg,
        )

    @classmethod
    def from_config(cls, cfg, config):
        # vision encoder
        visual_encoder_config = OmegaConf.to_container(cfg.timesformer)
        visual_encoder = TimeSformer(**visual_encoder_config)

        # text encoder
        text_encoder = XBertEncoder.from_config(cfg)
        text_encoder2 = XBertEncoder.from_config(cfg)

        audio_encoder = WhisperModel.from_pretrained(cfg.get("audio_encoder")).encoder

        model = cls(
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            text_encoder2=text_encoder2,
            audio_encoder=audio_encoder,
            config=config
        )

        num_patches = (
            visual_encoder_config["image_size"] // visual_encoder_config["patch_size"]
        ) ** 2
        num_frames = visual_encoder_config["n_frms"]

        model.load_checkpoint_from_config(
            cfg, num_frames=num_frames, num_patches=num_patches
        )

        return model
