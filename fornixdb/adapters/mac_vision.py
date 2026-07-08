"""Image-tower adapter — a local CLIP-family model that vectorizes frames for
the watch loop. A reference host adapter for `fornixdb.watchloop` on Apple
silicon.

The watch loop needs two things from an image model, and this one class serves
both from a single tower so their spaces agree:

  embed_image(frame) -> vector   the GATE lane (hot path): a fast per-frame
                                 vector BEFORE any file exists, so the salience
                                 gate can decide commit/hold. `frame` is a str
                                 path OR raw encoded image bytes (camera).
  embed_artifact(paths) -> [..]  the ModalEmbedder protocol (senses.ModalEmbedder):
                                 the LATENT lane, run only on committed keyframe
                                 files so modal_neighbors can find similar frames.

`name` keys every modal_embedding row, so it must be STABLE across runs — the
latent lane only ever compares vectors from one model's space.

Local-first and Apple-silicon-native: the default backend loads a standard
CLIP image tower on MLX from vendored Apple mlx-examples code (MIT, under
_vendor/mlx_examples_clip), with weights pulled from the permissively licensed
`mlx-community` CLIP repos on first use — all imported lazily, so importing
this module (and the whole test suite) needs no ML dependency. The model choice
is a one-function seam (`_clip_encoder`); any other local tower drops in through
the same `encode(image) -> vector` contract without touching the loop.
Everything here is generic to any Mac: no machine-specific state is encoded,
only the shape of the pipeline, so it is safe to share as an example other
users can fork.
"""
from __future__ import annotations

import io
import math
from typing import Callable

__all__ = ["ImageEmbedder", "clip_embedder"]


def _l2_normalize(vec: list[float]) -> list[float]:
    """Unit-length the vector so cosine in the gate is scale-free; a zero
    vector passes through unchanged (cosine() already treats it as 0.0)."""
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else list(vec)


def _default_open_image(frame):
    """Decode a watch-loop frame (str path OR encoded bytes) to a PIL RGB
    image. PIL is imported lazily so this module stays import-light; only the
    real embedding path pays for it."""
    from PIL import Image  # optional [mac] extra; lazy on purpose

    if isinstance(frame, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(frame)))
    elif isinstance(frame, str):
        img = Image.open(frame)
    else:
        raise TypeError(f"frame must be a path or encoded bytes, "
                        f"got {type(frame).__name__}")
    return img.convert("RGB")


class ImageEmbedder:
    """A ModalEmbedder over any local image tower.

    `encode(image) -> vector` is the whole model contract — it takes whatever
    `open_image` returns (a PIL image by default) and returns a raw vector; this
    class owns frame decoding, L2 normalization, and the two entry points the
    watch loop calls. Both `encode` and `open_image` are injectable so the
    class is unit-testable with no model and no PIL, the way the other adapters
    take injectable readers/clocks.
    """

    def __init__(self, encode: Callable[[object], list[float]], *, name: str,
                 open_image: Callable[[object], object] | None = None) -> None:
        self._encode = encode
        self.name = name
        self._open = open_image or _default_open_image

    def embed_image(self, frame) -> list[float]:
        """Gate-lane entry: vector for one frame (path or bytes), normalized."""
        return _l2_normalize(self._encode(self._open(frame)))

    def embed_artifact(self, paths: list[str]) -> list[list[float]]:
        """ModalEmbedder protocol: latent-lane vectors for committed keyframes."""
        return [self.embed_image(p) for p in paths]

    __call__ = embed_image      # so the instance is itself a gate `embed=`


def _remap_clip_key(k: str) -> str:
    """Adapt a key from the `mlx-community` CLIP checkpoints (an older
    mlx-examples conversion using `nn.TransformerEncoder`-style names and flat
    embeddings) to the current vendored model's layout. The mapping is a pure
    rename — same 398 tensors, same shapes — verified key-for-key at load time."""
    import re

    exact = {  # bare arrays that move under `.embeddings` (and gain `.weight`)
        "text_model.position_embedding":
            "text_model.embeddings.position_embedding.weight",
        "vision_model.position_embedding":
            "vision_model.embeddings.position_embedding.weight",
        "vision_model.class_embedding":
            "vision_model.embeddings.class_embedding",
    }
    if k in exact:
        return exact[k]

    k = k.replace(".attention.query_proj", ".self_attn.q_proj")
    k = k.replace(".attention.key_proj", ".self_attn.k_proj")
    k = k.replace(".attention.value_proj", ".self_attn.v_proj")
    k = k.replace(".attention.out_proj", ".self_attn.out_proj")
    k = k.replace(".linear1", ".mlp.fc1")
    k = k.replace(".linear2", ".mlp.fc2")
    k = k.replace(".ln1", ".layer_norm1")
    k = k.replace(".ln2", ".layer_norm2")
    k = re.sub(r"^(text_model|vision_model)\.layers\.",
               r"\1.encoder.layers.", k)
    k = k.replace("text_model.token_embedding.",
                  "text_model.embeddings.token_embedding.")
    k = k.replace("vision_model.patch_embedding.",
                  "vision_model.embeddings.patch_embedding.")
    k = k.replace("vision_model.pre_layernorm", "vision_model.pre_layrnorm")
    return k


def _load_clip(path: str):
    """Build the vendored CLIP model and load a Hugging Face MLX checkpoint from
    `path`, remapping the `mlx-community` key layout to the current model. We
    load here — not via the vendored `from_pretrained` — so that file stays
    byte-for-byte upstream. Raises if the remap doesn't reproduce the model's
    parameter set exactly, so a checkpoint drift fails loudly, never silently."""
    import glob
    import json
    from pathlib import Path

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from ._vendor.mlx_examples_clip.model import (CLIPConfig, CLIPModel,
                                                  CLIPTextConfig,
                                                  CLIPVisionConfig)

    p = Path(path)
    cfg = json.loads((p / "config.json").read_text())
    tc, vc = cfg["text_config"], cfg["vision_config"]
    model = CLIPModel(CLIPConfig(
        text_config=CLIPTextConfig(
            num_hidden_layers=tc["num_hidden_layers"],
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_attention_heads=tc["num_attention_heads"],
            max_position_embeddings=tc["max_position_embeddings"],
            vocab_size=tc["vocab_size"], layer_norm_eps=tc["layer_norm_eps"]),
        vision_config=CLIPVisionConfig(
            num_hidden_layers=vc["num_hidden_layers"],
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_attention_heads=vc["num_attention_heads"], num_channels=3,
            image_size=vc["image_size"], patch_size=vc["patch_size"],
            layer_norm_eps=vc["layer_norm_eps"]),
        projection_dim=cfg["projection_dim"]))

    raw = {}
    for wf in glob.glob(str(p / "*.safetensors")) + glob.glob(str(p / "*.npz")):
        raw.update(mx.load(wf))
    weights = {}
    for k, v in raw.items():
        if "position_ids" in k:            # a HF/pytorch artifact the model omits
            continue
        k = _remap_clip_key(k)
        # the conv weight must be mlx channels-last [out, kH, kW, in]; transpose
        # only a pytorch-layout [out, in, kH, kW] tensor (npz is already mlx).
        if k.endswith("patch_embedding.weight") and v.ndim == 4 and v.shape[1] == 3:
            v = v.transpose(0, 2, 3, 1)
        weights[k] = v

    expected = {k for k, _ in tree_flatten(model.parameters())}
    got = set(weights)
    if got != expected:
        raise RuntimeError(
            "CLIP checkpoint keys do not match the vendored model after remap "
            f"(missing={sorted(expected - got)[:4]}, "
            f"extra={sorted(got - expected)[:4]})")
    model.load_weights(list(weights.items()))
    return model


def _clip_encoder(model_id: str) -> Callable[[object], list[float]]:
    """Build an `encode(PIL image) -> vector` backed by a standard CLIP image
    tower on MLX. `model_id` is a Hugging Face repo of a MLX-format CLIP model
    (default `mlx-community/clip-vit-base-patch32`, Apache-2.0); its snapshot is
    downloaded once via `huggingface_hub` and cached. Imported lazily; raises a
    pointed error if the optional Mac extras are missing so the fix
    (`pip install 'fornixdb[mac]'`) is obvious.

    The image tower runs natively on Apple silicon. The CLIP code is vendored
    from Apple's mlx-examples (MIT); a different local tower drops in behind the
    same `encode(image) -> vector` contract without the loop noticing.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:                       # pragma: no cover - env-dep
        raise ImportError(
            "the watch embedder needs the optional Mac extras — install them: "
            "pip install 'fornixdb[mac]'  (adds mlx, huggingface-hub, numpy, "
            "pillow, opencv-python; all imported lazily, the core stays "
            "stdlib-only)"
        ) from e

    from ._vendor.mlx_examples_clip import CLIPImageProcessor

    path = snapshot_download(repo_id=model_id)
    model = _load_clip(path)
    processor = CLIPImageProcessor.from_pretrained(path)

    def encode(image) -> list[float]:
        pixel_values = processor([image])              # [1, H, W, C]
        feats = model.get_image_features(pixel_values)  # [1, projection_dim]
        return list(feats[0].tolist())

    return encode


def clip_embedder(model_id: str = "mlx-community/clip-vit-base-patch32",
                  *, encode: Callable[[object], list[float]] | None = None,
                  open_image: Callable[[object], object] | None = None,
                  ) -> ImageEmbedder:
    """The default watch embedder: a standard CLIP image tower on MLX, keyed by
    `model_id` (stable — it names the latent-lane space). Pass `encode=` to
    supply any other local tower (or a fake, in tests); otherwise the MLX
    backend is built lazily on the FIRST embed, so constructing the embedder
    (and importing this module) never downloads weights or loads a model."""
    if encode is None:
        _real: list = []                           # one-slot lazy cache

        def encode(image) -> list[float]:          # noqa: F811 - lazy default
            if not _real:
                _real.append(_clip_encoder(model_id))
            return _real[0](image)

    return ImageEmbedder(encode, name=model_id, open_image=open_image)
