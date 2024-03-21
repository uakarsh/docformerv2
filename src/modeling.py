from transformers import T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)
from transformers.utils import is_torch_fx_proxy
import torch

import torch.nn as nn
import collections.abc

## Ref:https://github.com/huggingface/transformers/blob/ff841900e45763114d2417fb24ce29d950c6c956/src/transformers/models/vit/modeling_vit.py#L146
class PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.d_model

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.max_image_tokens = config.max_image_tokens ## If we limit the max_image_tokens to a number, would it capture the global context?
        ## Should we keep the convolution kernel's size to be (16, 16) rather than just (2, 2), so sequence length can be reduced and we are
        ## able to capture global context?

        self.conv_projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.linear_projection = nn.Linear(hidden_size, hidden_size)

        self.positional_embedding = nn.Embedding(self.max_image_tokens, hidden_size)


    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.conv_projection(pixel_values).flatten(2).transpose(1, 2)
        embeddings = self.linear_projection(embeddings)[:, :self.max_image_tokens, :]

        positions = torch.arange(0, self.max_image_tokens).unsqueeze(0).to(embeddings.device)
        position_embedding = self.positional_embedding(positions)

        return embeddings + position_embedding


## Ref: https://github.com/uakarsh/latr/blob/1e73c1a99f9a0db4d85177259226148a65556069/src/new_latr/modeling.py#L11

class SpatialModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_left_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model // 2)
        self.bottom_right_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model // 2)
        self.top_left_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model // 2)
        self.bottom_right_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model // 2)
        self.width_emb = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        self.height_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)

    def forward(self, coordinates):

        top_left_x_feat = self.top_left_x(coordinates[:, :, 0])
        top_left_y_feat = self.top_left_y(coordinates[:, :, 1])
        bottom_right_x_feat = self.bottom_right_x(coordinates[:, :, 2])
        bottom_right_y_feat = self.bottom_right_y(coordinates[:, :, 3])
        width_feat = self.width_emb(coordinates[:, :, 2] - coordinates[:, :, 0])
        height_feat = self.height_emb(coordinates[:, :, 3] - coordinates[:, :, 1])

        layout_feature = torch.cat([top_left_x_feat, bottom_right_x_feat], axis = -1) + torch.cat([top_left_y_feat, bottom_right_y_feat], axis = -1) + \
             width_feat + height_feat
        return layout_feature

class DocFormerV2(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config=config)
        self.spatial_feat_extractor = SpatialModule(config)
        self.img_feature_extractor = PatchEmbeddings(config)
        self.modality_embedding = nn.Embedding(2, config.d_model)

    def forward(
            self,
            input_ids=None,
            bbox=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            pixel_values=None,
            labels=None,
            head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=True,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,) :

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            inputs_embeds, attention_mask = self.calculate_embedding(
                pixel_values, bbox, input_ids, attention_mask)
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = encoder_outputs[0]

        if decoder_input_ids == None:
            decoder_input_ids = self._shift_right(input_ids)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.config.d_model**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + \
                decoder_outputs[2:] + (encoder_outputs[0],) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "bbox": kwargs.get("bbox", None),
            "pixel_values": kwargs.get("pixel_values", None),
        }

    def calculate_embedding(self, img, bbox, input_ids, attention_mask):
        img_feat = self.img_feature_extractor(img)
        spatial_feat = self.spatial_feat_extractor(bbox)
        language_feat = self.shared(input_ids)

        layout_feat = spatial_feat + language_feat
        img_modality_token = self.modality_embedding(torch.zeros(1, img_feat.shape[1]).long().to(self.device))
        lang_modality_token = self.modality_embedding(torch.ones(1, language_feat.shape[1]).long().to(self.device))

        img_feat += img_modality_token
        layout_feat += lang_modality_token

        multi_modal_feat = torch.cat([img_feat, layout_feat], axis=1)
        input_attention_mask = torch.cat(
            [torch.ones(img_feat.shape[:2]).to(img_feat.device), attention_mask], axis=1)
        
        return multi_modal_feat, input_attention_mask