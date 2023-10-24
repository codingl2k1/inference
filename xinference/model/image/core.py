# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections.abc
import logging
import os
from typing import List, Optional, Tuple

from pydantic import BaseModel

from ...constants import XINFERENCE_CACHE_DIR
from ..core import ModelDescription
from .stable_diffusion.core import DiffusionModel

MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class ImageModelFamilyV1(BaseModel):
    model_family: str
    model_name: str
    model_id: str
    model_revision: str
    controlnet: Optional[List["ImageModelFamilyV1"]]


class ImageModelDescription(ModelDescription):
    def __init__(self, model_spec: ImageModelFamilyV1):
        self._model_spec = model_spec

    def to_dict(self):
        return {
            "model_type": "image",
            "model_name": self._model_spec.model_name,
            "model_family": self._model_spec.model_family,
            "model_revision": self._model_spec.model_revision,
            "controlnet": self._model_spec.controlnet,
        }


def match_diffusion(model_name: str) -> ImageModelFamilyV1:
    from . import BUILTIN_IMAGE_MODELS

    if model_name in BUILTIN_IMAGE_MODELS:
        return BUILTIN_IMAGE_MODELS[model_name]
    else:
        raise ValueError(
            f"Image model {model_name} not found, available"
            f"model list: {BUILTIN_IMAGE_MODELS.keys()}"
        )


def cache(model_spec: ImageModelFamilyV1):
    # TODO: cache from uri
    import huggingface_hub

    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
    )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    for current_attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            huggingface_hub.snapshot_download(
                model_spec.model_id,
                revision=model_spec.model_revision,
                local_dir=cache_dir,
                local_dir_use_symlinks=True,
                resume_download=True,
            )
            break
        except huggingface_hub.utils.LocalEntryNotFoundError:
            remaining_attempts = MAX_ATTEMPTS - current_attempt
            logger.warning(
                f"Attempt {current_attempt} failed. Remaining attempts: {remaining_attempts}"
            )
    else:
        raise RuntimeError(
            f"Failed to download model '{model_spec.model_name}' after {MAX_ATTEMPTS} attempts"
        )
    return cache_dir


def create_image_model_instance(
    model_uid: str, model_name: str, **kwargs
) -> Tuple[DiffusionModel, ImageModelDescription]:
    model_spec = match_diffusion(model_name)
    controlnet = kwargs.get("controlnet")
    # Handle controlnet
    if controlnet is not None:
        if isinstance(controlnet, str):
            controlnet = [controlnet]
        elif not isinstance(controlnet, collections.abc.Sequence):
            raise ValueError("controlnet should be a str or a list of str.")
        elif set(controlnet) != len(controlnet):
            raise ValueError("controlnet should be a list of unique str.")
        elif not model_spec.controlnet:
            raise ValueError(f"Model {model_name} has empty controlnet list.")

        controlnet_model_paths = []
        assert model_spec.controlnet is not None
        for name in controlnet:
            for cn_model_spec in model_spec.controlnet:
                if cn_model_spec.model_name == name:
                    model_path = cache(cn_model_spec)
                    controlnet_model_paths.append(model_path)
                    break
            else:
                raise ValueError(
                    f"controlnet `{name}` is not supported for model `{model_name}`."
                )
        if len(controlnet_model_paths) == 1:
            kwargs["controlnet"] = controlnet_model_paths[0]
        else:
            kwargs["controlnet"] = controlnet_model_paths
    model_path = cache(model_spec)
    model = DiffusionModel(model_uid, model_path, **kwargs)
    model_description = ImageModelDescription(model_spec)
    return model, model_description
