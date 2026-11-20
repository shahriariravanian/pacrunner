"""pacrunner.umami against the UMAMI goldens (tools/goldens: pac-julia's umami_libraries, nothing built):
for every golden set with an artifact's metadata, the provider's libraries and channels
(libraries.json), and per library its umami_tables.h, umami_program.h and compiled folders
(straight-line and interpreter), byte for byte; and one provider built on the CPU (cc).

The golden sets with metadata are the corpus's (PACRUNNER_GOLDENS, else ~/symjit-data/pacrunner-goldens):
skipped without them, or without symjit >= 2.29.

    python -m unittest discover -s test/python -p test_umami.py
"""

import ctypes
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from goldens import bits, first_difference, golden_sets

from pacrunner import codegen, uir, umami

try:
    import symjit
    HAVE_MIR_JSON = hasattr(symjit, "mir_json")
except ImportError:
    HAVE_MIR_JSON = False

UMAMI_SETS = [(n, d, e, m) for (n, d, e, m) in golden_sets() if m.get("metadata") and Path(m["metadata"]).is_file()]


def provider_of(export, metadata):
    d = uir.read_export(export)
    meta = umami.read_metadata(metadata, d)
    programs = umami.export_programs(d)
    (libraries, channels) = umami.provider_libraries(meta, programs, export)
    return d, meta, programs, libraries, channels


@unittest.skipUnless(UMAMI_SETS and HAVE_MIR_JSON, "no UMAMI goldens (the corpus) or no symjit >= 2.29")
class Goldens(unittest.TestCase):
    def test_libraries(self):
        for (name, folder, export, m) in UMAMI_SETS:
            with self.subTest(export=name):
                (d, meta, programs, libraries, channels) = provider_of(export, m["metadata"])
                ours = {"libraries": [{"index": k, "process": l.process, "colour": l.colour, "permutation": list(l.permutation),
                                       "colour_index": l.colour_index, "factor": bits(l.factor),
                                       "helicity_export": list(l.helicity_export)} for k, l in enumerate(libraries)],
                        "channels": [{"flavor_libraries": fl, "libraries": sorted(set(fl))} for fl in channels]}
                golden = json.loads((folder / "umami" / "libraries.json").read_text())
                self.assertIsNone(first_difference(golden, ours))
                parameters = umami.export_parameters(d)
                for k, lib in enumerate(libraries):
                    g = folder / "umami" / f"{k:04d}"
                    l = umami.permuted(uir.optimize(programs[(lib.process, lib.colour)][0]), lib.permutation)
                    (tables, program) = umami.umami_program_h(lib, l, meta["provider"], len(meta["helicities"]), parameters)
                    self.assertEqual(tables, (g / "umami_tables.h").read_text(), f"library {k}")
                    self.assertEqual(program, (g / "umami_program.h").read_text(), f"library {k}")
                    for variant, interpreter in (("straight", False), ("interpreter", True)):
                        with tempfile.TemporaryDirectory() as out:
                            codegen.compile(out, l, interpreter=interpreter)
                            for f in ("runner.c", "runner.h", "Makefile"):
                                self.assertEqual((Path(out) / f).read_bytes(), (g / variant / f).read_bytes(),
                                                 f"library {k} {variant} {f}")

    @unittest.skipUnless(shutil.which("cc"), "no cc")
    def test_cpu_build(self):
        (name, folder, export, m) = min(UMAMI_SETS, key=lambda s: s[3]["umami_libraries"])
        with tempfile.TemporaryDirectory() as out:
            table = json.loads(Path(umami.umami_libraries(export, m["metadata"], out, quiet=True)).read_text())
            self.assertEqual(table["device"], "cpu")
            self.assertEqual(len(table["libraries"]), m["umami_libraries"])
            for entry in table["libraries"]:
                so = ctypes.CDLL(str(Path(out) / entry["path"]))
                for symbol in ("umami_get_meta", "umami_initialize", "umami_matrix_element", "umami_free"):
                    self.assertTrue(hasattr(so, symbol), symbol)
                with self.assertRaises(AttributeError):   # every other symbol hidden
                    so.run


class Units(unittest.TestCase):
    def test_permuted(self):
        l = uir.load_recurrence_export(Path(__file__).parents[1] / "data" / "export" / "zggg_union.json")[0]
        p = list(reversed(range(l.legs)))
        crossed = umami.permuted(l, p)
        if l.crossing is None:
            self.assertEqual(crossed.crossing, tuple((p[k], 1.0) for k in range(l.legs)))
        else:
            self.assertEqual(crossed.crossing, tuple((p[leg], s) for (leg, s) in l.crossing))
        with self.assertRaises(umami.UmamiError):
            umami.permuted(l, [0] * l.legs)

    def test_devices(self):
        self.assertEqual(umami.device_name("cuda_full"), "cuda_provider")
        with self.assertRaises(umami.UmamiError):
            umami.build_umami_library("/nonexistent", None, None, {}, 1, [], device="tpu")
        with self.assertRaises(umami.UmamiError):
            umami.build_umami_library("/nonexistent", None, None, {}, 1, [], precision="mixed")


if __name__ == "__main__":
    unittest.main()
