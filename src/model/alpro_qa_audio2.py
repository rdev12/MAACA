"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from warnings import warn

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import BertTokenizer
from src.model.alpro_outputs import (
    AlproIntermediateOutput,
    AlproOutputWithLogits,
)
from src.model.med import XBertEncoder
from src.model.timesformer.vit import TimeSformer
from torch import nn
import math
from transformers import WhisperModel

class LSTM_fc(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_seq_len, output_size):
        super(LSTM_fc, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=2*hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Select the output from the last time step
        # output = out[:, -1, :].clone()  # Clone the tensor to make it out of place operation
        # print(out[:, -1, :].shape)
        
        output = self.fc(out[:, :self.output_seq_len, :])

        return output

class Gate_Attention(nn.Module):
    def __init__(self, num_hidden_a, num_hidden_b, num_hidden):
        super(Gate_Attention, self).__init__()
        self.hidden = num_hidden
        self.w1 = nn.Parameter(torch.Tensor(num_hidden_a, num_hidden))
        self.w2 = nn.Parameter(torch.Tensor(num_hidden_b, num_hidden))
        self.bias = nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()

    def reset_parameter(self):
        stdv1 = 1. / math.sqrt(self.hidden)
        stdv2 = 1. / math.sqrt(self.hidden)
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, a, b):
        wa = torch.matmul(a, self.w1)
        wb = torch.matmul(b, self.w2)
        gated = wa + wb + self.bias
        gate = torch.sigmoid(gated)
        output = gate * a + (1 - gate) * b
        return output  # Clone the tensor to make it out of place operation

 
class FC_head(nn.Module):
    def __init__(self, num_classes, hidden_dim, llm_embed_dim, add_pooling = False):
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
        
class MAACA(nn.Module):
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
        targets = aspect if self.model_config.num_classes == 7 else complaint

        # # forward text
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
        # print("visual input shape", visual_inputs.shape)
        # import pdb; pdb.set_trace()

        video_embeds = self.visual_encoder.forward_features(visual_inputs)
        video_embeds = self.transform_video_to_hidden(video_embeds)

        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        # # forward cross-encoder
        attention_mask1 = torch.cat([text.attention_mask, video_atts], dim=1)
        embedding_output1 = torch.cat([text_embeds, video_embeds], dim=1)

        # import pdb; pdb.set_trace()
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

        # import pdb; pdb.set_trace()
        fused_output = self.gate_fusion(encoder_output1.last_hidden_state, encoder_output2.last_hidden_state)

        # prediction = self.classifier(fused_output)
        complaint_prediction = self.complaint_head(fused_output)
        aspect_prediction = self.aspect_head(fused_output)

        if is_train:
            # import pdb; pdb.set_trace()
            # print(prediction)
            complaint_loss = F.cross_entropy(complaint_prediction, complaint)
            aspect_loss = F.cross_entropy(aspect_prediction, aspect)
            loss = complaint_loss + aspect_loss

            # return {"loss": loss}
            return AlproOutputWithLogits(
                loss=complaint_loss if self.model_config.num_classes == 2 else aspect_loss,
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
        
        # for key, value in visual_encoder_config.items():
        #     visual_encoder_config[key] = getattr(config, key)

        visual_encoder = TimeSformer(**visual_encoder_config)

        # text encoder
        text_encoder = XBertEncoder.from_config(cfg)
        text_encoder2 = XBertEncoder.from_config(cfg)
        text_encoder3 = XBertEncoder.from_config(cfg)

        # audio encoder
        audio_encoder = WhisperModel.from_pretrained(cfg.get("audio_encoder")).encoder

        # num_classes = cfg.get("num_classes", -1)
        # hidden_size = cfg.get("hidden_size", 768)
        # max_txt_len = cfg.get("max_txt_len", 128)
        # hidden_size=hidden_size,
        # num_classes=num_classes,
        # max_txt_len=max_txt_len,

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
