import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

import timm
from transformers import BertConfig, BertModel, BertForSequenceClassification

from model_utils import Identity

class Trained_CLIP_TextEncoder(nn.Module):
    def __init__(self, trainable: bool = False) -> None:
        super().__init__()

        # ファイル全体をリストとして読み込み
        with open("information/vocabulary_cjkvi_fixed.txt") as f:
            vocabulary_list = [s.strip() for s in f.readlines()]
        vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

        image_encoder_alias = "resnet50"
        model = CLIPDualEncoderModel(image_encoder_alias, len(vocabulary_list))
        model_path = "pretraind_weight/CLIP.pt"
        weight = torch.load(model_path)
        model.load_state_dict(weight)

        for param in model.parameters():
            param.requires_grad = trainable

        self.encoder = model.text_encoder
        self.text_projection = model.text_projection

    def forward(self, input_ids):
        text_features = self.encoder(input_ids)
        text_embeddings = self.text_projection(text_features)

        return text_embeddings

class Trained_CLIP_ImageEncoder(nn.Module):
    def __init__(self, trainable: bool = False) -> None:
        super().__init__()

        # ファイル全体をリストとして読み込み
        with open("information/vocabulary_cjkvi_fixed.txt") as f:
            vocabulary_list = [s.strip() for s in f.readlines()]
        vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

        image_encoder_alias = "resnet50"
        model = CLIPDualEncoderModel(image_encoder_alias, len(vocabulary_list))
        model_path = "pretraind_weight/CLIP.pt"
        weight = torch.load(model_path)
        model.load_state_dict(weight)

        for param in model.parameters():
            param.requires_grad = trainable

        self.encoder = model.image_encoder
        self.image_projection = model.image_projection

    def forward(self, input_ids):
        image_features = self.encoder(input_ids)
        image_embeddings = self.image_projection(image_features)

        return image_embeddings

class CLIPDualEncoderModel(nn.Module):
    def __init__(
        self,
        image_encoder_alias: str,
        vocabulary_size: int,
        image_encoder_pretrained: bool = True,
        image_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        image_embedding_dims: int = 2048,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_alias,
            pretrained=image_encoder_pretrained,
            trainable=image_encoder_trainable,
        )
        self.text_encoder = TextEncoder(
            vocabulary_size=vocabulary_size, trainable=text_encoder_trainable
        )
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

    def forward(self, image, input_ids):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)

        x += projected

        return self.layer_norm(x)

class ImageEncoder(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool = True, trainable: bool = True
    ) -> None:
        super().__init__()

        if "resnet" in model_name:
            self.model = timm.create_model(
                model_name, pretrained=True, num_classes=3057, global_pool="avg"
            )
            # model_path = "ImageEncoder/result_model:resnet50_m:2_s:224/model/best_classifier_model.pt"
            # weight = torch.load(model_path, map_location=device)
            # self.model.load_state_dict(weight)

            self.model.fc = Identity()
        elif "vit" in model_name:
            self.model = timm.create_model(
                model_name, pretrained=True, num_classes=3057
            )
            # model_path = "ImageEncoder/result_model:vit_base_patch8_224_m:2_s:224/model/best_classifier_model.pt"
            # weight = torch.load(model_path, map_location=device)
            # self.model.load_state_dict(weight)

            self.model.head = Identity()

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, vocabulary_size: int, trainable: bool = True) -> None:
        super().__init__()

        # Initializing a BERT bert-base-uncased style configuration
        configuration = BertConfig(vocab_size = vocabulary_size,
                                max_position_embeddings=13,
                                hidden_size=768)

        model_ckpt = "yosuke/bert-base-japanese-char"
        self.model = BertForSequenceClassification(configuration).from_pretrained(model_ckpt, num_labels=3057)
        bertembeddings = BertEmbeddings(configuration)
        self.model.bert.embeddings = bertembeddings

        self.model.dropout = Identity()
        self.model.classifier = Identity()

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    # def forward(self, input_ids):
    #     outputs = self.model(**input_ids)
    #     last_hidden_states = outputs[0]

    #     return last_hidden_states[:,self.target_token_idx,:]

    def forward(self, input_ids):
        outputs = self.model(**input_ids).logits
        return outputs

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings