"""Test package.

Force vectors OFF for the whole suite so tests are deterministic whether or not
model2vec is installed in the environment (it is now a default dependency, so a
real model would otherwise auto-embed and perturb keyword-only expectations).
Tests that exercise vectors pass an explicit embedder or clear this switch
locally — the env override only gates the AUTO path."""

import os

os.environ["FORNIXDB_VECTORS"] = "off"
