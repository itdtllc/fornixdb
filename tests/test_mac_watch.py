"""Watch-loop P2: the Mac frame-source and image-embedder adapters, and the
`fornixdb watch` CLI wiring. Everything here runs with NO camera, NO screen,
and NO ML dependency — `grab`, `clock`, `sleep`, `encode`, and `open_image`
are all injectable, and the CLI test patches the adapters with fakes, so the
whole loop is exercised deterministically the way test_watchloop already does.
"""
import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fornixdb.adapters import mac_camera, mac_vision
from fornixdb.cli import main


class TestFramePacing(unittest.TestCase):
    """`_stream` turns a grab() into paced (timestamp, frame) pairs."""

    def test_stops_when_grabber_ends_and_sleeps_between(self):
        frames = [b"a", b"b", b"c"]
        it = iter(frames)
        clocks = iter([10.0, 11.0, 12.0])
        slept = []
        out = list(mac_camera._stream(
            lambda: next(it, None), rate_hz=2.0, count=None,
            clock=lambda: next(clocks), sleep=slept.append))
        self.assertEqual([f for _, f in out], frames)
        self.assertEqual([t for t, _ in out], [10.0, 11.0, 12.0])
        self.assertEqual(slept, [0.5, 0.5, 0.5])     # paces after each grab

    def test_count_stops_at_the_limit(self):
        it = iter([b"a", b"b", b"c", b"d"])
        slept = []
        out = list(mac_camera._stream(
            lambda: next(it, None), rate_hz=1.0, count=2,
            clock=lambda: 0.0, sleep=slept.append))
        self.assertEqual(len(out), 2)                # count reached, source not exhausted
        self.assertEqual(slept, [1.0])               # one pace between the two frames

    def test_empty_source_yields_nothing(self):
        out = list(mac_camera._stream(
            lambda: None, rate_hz=0, count=None, clock=lambda: 0.0,
            sleep=lambda _s: None))
        self.assertEqual(out, [])

    def test_camera_frames_accepts_injected_grab(self):
        it = iter([b"x", b"y"])
        out = list(mac_camera.camera_frames(
            grab=lambda: next(it, None), rate_hz=0, count=2,
            clock=lambda: 1.0, sleep=lambda _s: None))
        self.assertEqual([f for _, f in out], [b"x", b"y"])


class TestOpenStream(unittest.TestCase):
    """Source-string dispatch. camera/screen return generators without touching
    cv2 or the screen (the body only runs on iteration); a bad file is caught."""

    def test_labels(self):
        _, label = mac_camera.open_stream("camera")
        self.assertEqual(label, "camera")
        _, label = mac_camera.open_stream("screen")
        self.assertEqual(label, "screen")

    def test_missing_file_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            mac_camera.open_stream("/no/such/video.mov")

    def test_existing_file_labels_as_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mov") as f:
            _, label = mac_camera.open_stream(f.name)
        self.assertEqual(label, "file")


class TestImageEmbedder(unittest.TestCase):
    """ImageEmbedder owns decode + L2-normalize; encode/open_image inject in."""

    def _emb(self, encode):
        return mac_vision.ImageEmbedder(encode, name="t", open_image=lambda f: f)

    def test_embed_image_is_unit_length(self):
        v = self._emb(lambda img: [3.0, 4.0]).embed_image("x")
        self.assertEqual(v, [0.6, 0.8])
        self.assertAlmostEqual(math.hypot(*v), 1.0)

    def test_zero_vector_passes_through(self):
        self.assertEqual(self._emb(lambda img: [0.0, 0.0]).embed_image("x"),
                         [0.0, 0.0])

    def test_call_matches_embed_image(self):
        emb = self._emb(lambda img: [1.0, 0.0])
        self.assertEqual(emb("x"), emb.embed_image("x"))

    def test_open_image_sees_bytes_and_paths(self):
        seen = []
        emb = mac_vision.ImageEmbedder(
            lambda img: [1.0, 0.0], name="t",
            open_image=lambda f: (seen.append(f), f)[1])
        emb.embed_image(b"raw")
        emb.embed_image("/frame.jpg")
        self.assertEqual(seen, [b"raw", "/frame.jpg"])

    def test_embed_artifact_maps_over_paths(self):
        emb = self._emb(lambda img: [3.0, 4.0])
        self.assertEqual(emb.embed_artifact(["a", "b"]), [[0.6, 0.8], [0.6, 0.8]])

    def test_factory_keeps_model_id_as_name_without_loading_mlx(self):
        emb = mac_vision.clip_embedder(
            "mlx-community/clip-vit-base-patch32", encode=lambda img: [1.0, 0.0],
            open_image=lambda f: f)
        self.assertEqual(emb.name, "mlx-community/clip-vit-base-patch32")
        self.assertEqual(emb.embed_image("x"), [1.0, 0.0])   # stable latent key


class TestClipKeyRemap(unittest.TestCase):
    """_remap_clip_key adapts the mlx-community CLIP checkpoint layout to the
    vendored model. It is a pure rename (no mlx needed); these lock the rules so
    a future edit can't silently break the load. The runtime already asserts the
    remapped set matches the model exactly — this pins the mapping itself."""

    def test_layer_stack_names(self):
        r = mac_vision._remap_clip_key
        self.assertEqual(
            r("vision_model.layers.0.attention.query_proj.weight"),
            "vision_model.encoder.layers.0.self_attn.q_proj.weight")
        self.assertEqual(
            r("vision_model.layers.11.attention.key_proj.bias"),
            "vision_model.encoder.layers.11.self_attn.k_proj.bias")
        self.assertEqual(
            r("text_model.layers.3.attention.value_proj.weight"),
            "text_model.encoder.layers.3.self_attn.v_proj.weight")
        self.assertEqual(
            r("text_model.layers.3.attention.out_proj.weight"),
            "text_model.encoder.layers.3.self_attn.out_proj.weight")
        self.assertEqual(
            r("vision_model.layers.5.linear1.bias"),
            "vision_model.encoder.layers.5.mlp.fc1.bias")
        self.assertEqual(
            r("vision_model.layers.5.linear2.weight"),
            "vision_model.encoder.layers.5.mlp.fc2.weight")
        self.assertEqual(
            r("vision_model.layers.5.ln1.weight"),
            "vision_model.encoder.layers.5.layer_norm1.weight")
        self.assertEqual(
            r("vision_model.layers.5.ln2.bias"),
            "vision_model.encoder.layers.5.layer_norm2.bias")

    def test_embeddings_regrouped(self):
        r = mac_vision._remap_clip_key
        self.assertEqual(r("vision_model.patch_embedding.weight"),
                         "vision_model.embeddings.patch_embedding.weight")
        self.assertEqual(r("text_model.token_embedding.weight"),
                         "text_model.embeddings.token_embedding.weight")
        # bare arrays that move under .embeddings (position gains .weight)
        self.assertEqual(r("vision_model.position_embedding"),
                         "vision_model.embeddings.position_embedding.weight")
        self.assertEqual(r("text_model.position_embedding"),
                         "text_model.embeddings.position_embedding.weight")
        self.assertEqual(r("vision_model.class_embedding"),
                         "vision_model.embeddings.class_embedding")

    def test_pre_layernorm_spelling(self):
        # the vendored model spells it `pre_layrnorm`
        r = mac_vision._remap_clip_key
        self.assertEqual(r("vision_model.pre_layernorm.weight"),
                         "vision_model.pre_layrnorm.weight")
        self.assertEqual(r("vision_model.pre_layernorm.bias"),
                         "vision_model.pre_layrnorm.bias")

    def test_unchanged_keys_pass_through(self):
        r = mac_vision._remap_clip_key
        for k in ("logit_scale", "text_projection.weight",
                  "visual_projection.weight", "vision_model.post_layernorm.weight",
                  "text_model.final_layer_norm.bias"):
            self.assertEqual(r(k), k)

    def test_remap_is_collision_free_over_the_real_keyset(self):
        # a full clip-vit-base-patch32 key set (12 text + 12 vision layers +
        # the non-layer keys) must remap 1:1 with no two keys colliding.
        r = mac_vision._remap_clip_key
        keys = ["logit_scale", "text_projection.weight", "visual_projection.weight",
                "text_model.token_embedding.weight", "text_model.position_embedding",
                "text_model.final_layer_norm.weight", "text_model.final_layer_norm.bias",
                "vision_model.class_embedding", "vision_model.patch_embedding.weight",
                "vision_model.position_embedding",
                "vision_model.pre_layernorm.weight", "vision_model.pre_layernorm.bias",
                "vision_model.post_layernorm.weight", "vision_model.post_layernorm.bias"]
        for tv, n in (("text_model", 12), ("vision_model", 12)):
            for i in range(n):
                for sub in ("attention.query_proj", "attention.key_proj",
                            "attention.value_proj", "attention.out_proj",
                            "linear1", "linear2", "ln1", "ln2"):
                    for part in ("weight", "bias"):
                        keys.append(f"{tv}.layers.{i}.{sub}.{part}")
        out = [r(k) for k in keys]
        self.assertEqual(len(out), len(set(out)))     # bijective, no collisions
        self.assertEqual(len(set(keys)), len(set(out)))


def _fake_encode(x):
    """Gate lane sees bytes (first byte picks an axis); the latent lane sees a
    committed keyframe PATH (str) — any stable vector is fine there."""
    if isinstance(x, str):
        return [0.0, 0.0, 1.0]
    first = bytes(x)[:1]
    return {b"A": [1.0, 0.0, 0.0], b"B": [0.0, 1.0, 0.0]}.get(first, [0.0, 0.0, 0.5])


class TestWatchCli(unittest.TestCase):
    """`fornixdb watch` end to end with the camera/screen and the MLX model
    faked out: a scripted frame list drives the real gate + watchloop + see()."""

    def _run(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(list(argv))
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_scene_change_commits_two_see_memories(self):
        frames = [(0.0, b"AAAA"), (1.0, b"AAAA"),   # first commits, then quiet
                  (2.0, b"BBBB"), (3.0, b"BBBB")]    # A->B scene change commits
        fake_emb = mac_vision.ImageEmbedder(
            _fake_encode, name="fake_clip", open_image=lambda f: f)

        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            with mock.patch.object(mac_camera, "open_stream",
                                   return_value=(iter(frames), "screen")), \
                 mock.patch.object(mac_vision, "clip_embedder",
                                   return_value=fake_emb):
                out = self._run("--db", db, "--no-shared", "watch",
                                "--source", "screen", "--seconds", "100")
            self.assertIn("watch[screen]: session start", out)
            self.assertIn("watch[screen]: scene change", out)
            self.assertIn("2 memories committed.", out)
            # the committed frames are recallable as ordinary sight memories
            found = self._run("--db", db, "--no-shared", "recall", "scene change")
        self.assertIn("watch[screen]: scene change", found)


class TestLookCli(unittest.TestCase):
    """`fornixdb look` — one synchronous glance, camera + VLM faked."""

    def _run(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(list(argv))
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def _faked(self, caption):
        return (mock.patch.object(
                    mac_camera, "open_stream",
                    return_value=(iter([(0.0, b"AAAA")]), "camera")),
                mock.patch.object(mac_vision, "vlm_captioner",
                                  return_value=lambda p: caption))

    def test_prints_the_caption_and_remembers_nothing_by_default(self):
        cam, vlm = self._faked("an older man in a chair")
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            with cam, vlm:
                out = self._run("--db", db, "--no-shared", "look",
                                "--source", "camera")
            self.assertIn("an older man in a chair", out)
            found = self._run("--db", db, "--no-shared", "recall", "older man")
        self.assertNotIn("an older man in a chair", found)   # ephemeral

    def test_remember_stores_a_recallable_see_memory(self):
        cam, vlm = self._faked("a red mug on the desk")
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "m.db")
            with cam, vlm:
                self._run("--db", db, "--no-shared", "look", "--source",
                          "camera", "--remember")
            found = self._run("--db", db, "--no-shared", "recall", "red mug")
        self.assertIn("a red mug on the desk", found)


if __name__ == "__main__":
    unittest.main()
