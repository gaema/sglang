"""CPU unit tests for finding safetensors shards that the index places below the model root.

nvidia/Qwen3.8-Flash-Next-FP4-Local keeps its shards under ``tensors/`` and its
``model.safetensors.index.json`` names them by that relative path. With no
``*.safetensors`` at the root the loader must take the shard list from the
index; a root that has shards keeps the glob.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest

from sglang.srt.model_loader.loader import indexed_safetensors_files
from sglang.test.test_utils import CustomTestCase

INDEX = "model.safetensors.index.json"


def _layout(root, shards, index_names):
    for s in shards:
        os.makedirs(os.path.dirname(os.path.join(root, s)) or root, exist_ok=True)
        open(os.path.join(root, s), "wb").close()
    if index_names is not None:
        with open(os.path.join(root, INDEX), "w") as f:
            json.dump({"weight_map": {"w%d" % i: n for i, n in enumerate(index_names)}}, f)


class TestIndexedSafetensorsFiles(CustomTestCase):
    def test_subdirectory_shards_come_from_the_index(self):
        with tempfile.TemporaryDirectory() as d:
            names = ["tensors/model-00001-of-00002.safetensors",
                     "tensors/model-00002-of-00002.safetensors"]
            _layout(d, names, names + [names[0]])
            got = indexed_safetensors_files(d, INDEX, ["*.safetensors"])
            self.assertEqual(got, [os.path.join(d, n) for n in names])

    def test_nothing_without_an_index_a_present_shard_or_a_safetensors_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            _layout(d, ["tensors/a.safetensors"], None)
            self.assertEqual(indexed_safetensors_files(d, INDEX, ["*.safetensors"]), [])
        with tempfile.TemporaryDirectory() as d:
            _layout(d, [], ["tensors/missing.safetensors"])
            self.assertEqual(indexed_safetensors_files(d, INDEX, ["*.safetensors"]), [])
        with tempfile.TemporaryDirectory() as d:
            _layout(d, ["tensors/a.safetensors"], ["tensors/a.safetensors"])
            self.assertEqual(indexed_safetensors_files(d, INDEX, ["*.bin"]), [])


if __name__ == "__main__":
    unittest.main()
