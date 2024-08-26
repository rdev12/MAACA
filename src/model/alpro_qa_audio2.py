"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import BertTokenizer
from transformers import WhisperModel

from src.model.alpro_outputs import AlproIntermediateOutput, AlproOutputWithLogits
from src.model.med import XBertEncoder
from src.model.timesformer.vit import TimeSformer
from src.model.gate_fusion import Gate_Attention
from src.model.lstm_compression import LSTM_fc
from src.model.base_model import BaseModel

class FC_head(nn.Module):
    def __init__(self, num_classes, hidden_dim, llm_embed_dim, add_pooling=False):
        super(FC_head, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.llm_embed_dim = llm_embed_dim
        self.add_pooling = add_pooling

        self.pooling = nn.Linear(in_features=llm_embed_dim, out_features=llm_embed_dim)
        self.activation = nn.Tanh()

        self.fc1 = nn.Linear(in_features=llm_embed_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x):
        if self.add_pooling:
            x = self.pooling(x)
            x = self.activation(x)

        x = torch.mean(x, dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MAACA(BaseModel):
    def __init__(
        self, visual_encoder, text_encoder, text_encoder2, audio_encoder, config,
    ):
        super().__init__()
        self.model_config = config
         
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.visual_encoder = visual_encoder

        self.text_encoder = text_encoder
        self.text_encoder2 = text_encoder2

        self.audio_encoder = audio_encoder

        self.transform_audio_to_hidden = LSTM_fc(input_size=768, hidden_size=self.model_config.audio_transform_hidden_dim, num_layers=self.model_config.audio_transform_num_layers, output_seq_len=self.model_config.audio_output_seq_len, output_size=self.model_config.audio_transform_output_dim)
        self.transform_video_to_hidden = LSTM_fc(input_size=768, hidden_size=self.model_config.video_transform_hidden_dim, num_layers=self.model_config.video_transform_num_layers, output_seq_len=self.model_config.video_output_seq_len, output_size=self.model_config.video_transform_output_dim)

        self.gate_fusion = Gate_Attention(num_hidden_a=self.model_config.audio_transform_output_dim, num_hidden_b=self.model_config.video_transform_output_dim, num_hidden=self.model_config.fusion_output_dim)
        self.aspect_head = FC_head(num_classes=7, hidden_dim=self.model_config.linear_layer_hidden_dim, llm_embed_dim=self.model_config.fusion_output_dim, add_pooling=self.model_config.add_pooling)
        self.complaint_head = FC_head(num_classes=2, hidden_dim=self.model_config.linear_layer_hidden_dim, llm_embed_dim=self.model_config.fusion_output_dim, add_pooling=self.model_config.add_pooling)

        self.max_txt_len = self.model_config.max_txt_len

    def forward(self, video, audio, text_input, aspect, complaint, is_train=True):
        visual_inputs = video
        audio_inputs = audio
        question = text_input

        text = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_output = self.text_encoder.forward_text(
            text,
            token_type_ids=torch.zeros(
                text.input_ids.shape, dtype=torch.long, device=self.device
            ),
        )
        text_embeds = text_output.last_hidden_state # (b, text, 768)

        text_output2 = self.text_encoder2.forward_text(
            text,
            token_type_ids=torch.zeros(
                text.input_ids.shape, dtype=torch.long, device=self.device
            ),
        )
        text_embeds2 = text_output2.last_hidden_state # (b, text, 768)

        # forward visual
        # timeSformer asks for (b, c, t, h, w) as input.

        video_embeds = self.visual_encoder.forward_features(visual_inputs)
        video_embeds = self.transform_video_to_hidden(video_embeds)

        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        # # forward cross-encoder
        attention_mask1 = torch.cat([text.attention_mask, video_atts], dim=1)
        embedding_output1 = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_output1 = self.text_encoder(
            encoder_embeds=embedding_output1,
            attention_mask=attention_mask1,
            return_dict=True,
            mode="fusion",
        )

        audio_embeds = self.audio_encoder(audio_inputs).last_hidden_state # b, 1500, 768
        audio_embeds = self.transform_audio_to_hidden(audio_embeds) # b, 128, 768
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        attention_mask2 = torch.cat([text.attention_mask, audio_atts], dim=1)
        embedding_output2 = torch.cat([text_embeds2, audio_embeds], dim=1)

        encoder_output2 = self.text_encoder2(
            encoder_embeds=embedding_output2,
            attention_mask=attention_mask2,
            return_dict=True,
            mode="fusion",
        )

        fused_output = self.gate_fusion(encoder_output1.last_hidden_state, encoder_output2.last_hidden_state)

        complaint_prediction = self.complaint_head(fused_output)
        aspect_prediction = self.aspect_head(fused_output)

        if is_train:
            complaint_loss = F.cross_entropy(complaint_prediction, complaint)
            aspect_loss = F.cross_entropy(aspect_prediction, aspect)
            loss = complaint_loss + aspect_loss

            return AlproOutputWithLogits(
                loss=loss,
                intermediate_output=AlproIntermediateOutput(
                    video_embeds=video_embeds,
                    text_embeds=audio_embeds,
                    # encoder_output=[encoder_output1, encoder_output2],
                    encoder_output=encoder_output2,
                ),
                logits=(complaint_prediction, aspect_prediction),
            )
        else:
            return {"predictions": (complaint_prediction, aspect_prediction), "targets": (complaint, aspect)}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg, config):
        # vision encoder
        visual_encoder_config = OmegaConf.to_container(cfg.timesformer)
        visual_encoder = TimeSformer(**visual_encoder_config)

        # text encoder
        text_encoder = XBertEncoder.from_config(cfg)
        text_encoder2 = XBertEncoder.from_config(cfg)

        # audio encoder
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
