"""pacrunner.umami against the UMAMI goldens (tools/goldens: pac-julia's umami_libraries, nothing built):
for every golden set with an artifact's metadata, the provider's libraries and channels
(libraries.json), and per library its umami_tables.h, umami_program.h and compiled folders
(straight-line, interpreter and compile_metal's), byte for byte; and one provider built on the CPU (cc).

The golden sets with metadata are the corpus's (PACRUNNER_GOLDENS, else ~/symjit-data/pacrunner-goldens):
skipped without them, or without symjit >= 2.29.

    python -m unittest discover -s test/python -p test_umami.py
"""

import ctypes
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from goldens import bits, first_difference, golden_sets

from pacrunner import codegen, kernels, uir, umami

HAVE_MIR_JSON = kernels.symjit_problem() is None   # symjit >= 2.29 (its native library's version)

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
                    if (g / "metal").is_dir():   # the Metal build's generated files
                        with tempfile.TemporaryDirectory() as out:
                            codegen.compile_metal(out, l)
                            for f in ("runner.metal", "program.bin", "runner_metal.c", "runner_metal.h", "Makefile"):
                                self.assertEqual((Path(out) / f).read_bytes(), (g / "metal" / f).read_bytes(),
                                                 f"library {k} metal {f}")

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

    def test_an_export_already_read(self):
        (name, folder, export, m) = min(UMAMI_SETS, key=lambda s: s[3]["umami_libraries"])
        with tempfile.TemporaryDirectory() as out:   # `true` for cc: the sources only
            table = json.loads(Path(umami.umami_libraries(uir.read_export(export), m["metadata"], out, cc="true",
                                                          quiet=True)).read_text())
            self.assertIsNone(table["export"])
            self.assertEqual(len(table["libraries"]), m["umami_libraries"])

    def test_the_first_failure_stops_the_build(self):
        (name, folder, export, m) = max(UMAMI_SETS, key=lambda s: s[3]["umami_libraries"])
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(umami.UmamiError):
                umami.umami_libraries(export, m["metadata"], out, cc="false", jobs=1, quiet=True)
            built = list((Path(out) / "programs").iterdir())
            self.assertLess(len(built), m["umami_libraries"])   # the pending ones cancelled


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

    def test_replace_copy_keeps_an_open_file(self):
        with tempfile.TemporaryDirectory() as d:
            (src, dst) = (Path(d) / "new", Path(d) / "lib.so")
            src.write_bytes(b"new")
            dst.write_bytes(b"old")
            with open(dst, "rb") as held:   # a process that has the library loaded
                umami._replace_copy(src, dst)
                self.assertEqual(held.read(), b"old")
            self.assertEqual(dst.read_bytes(), b"new")
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["lib.so", "new"])

    def test_find_cc_takes_arguments(self):
        with mock.patch.dict(os.environ, {"CC": "env cc -m64"}):
            self.assertEqual(umami.find_cc(), "env cc -m64")
        with mock.patch.dict(os.environ, {"CC": "no-such-compiler-xyz -O2"}):
            self.assertIsNone(umami.find_cc())

    def test_errors_before_any_work(self):
        with self.assertRaises(umami.UmamiError):   # fmad: before the export is read or nvcc runs
            umami.umami_libraries("/nonexistent", "/nonexistent", "/nonexistent", device="cuda", fmad=True)

    def test_an_old_symjit_is_refused_first(self):
        with mock.patch("pacrunner.kernels.mir.symjit_version", return_value="2.28.3"):
            with self.assertRaises(kernels.MirError):   # before the export is read
                umami.umami_libraries("/nonexistent", "/nonexistent", "/nonexistent")

    def test_cli_error_is_exit_1(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(umami.main(["/nonexistent.json", "/nonexistent.json", "/nonexistent", "--fmad"]), 1)
        self.assertIn("fmad", err.getvalue())

    def test_devices(self):
        self.assertEqual(umami.device_name("cuda_full"), "cuda_provider")
        with self.assertRaises(umami.UmamiError):
            umami.build_umami_library("/nonexistent", None, None, {}, 1, [], device="tpu")
        with self.assertRaises(umami.UmamiError):
            umami.build_umami_library("/nonexistent", None, None, {}, 1, [], precision="mixed")
        # gpu: this system's GPU; metal builds on macOS only, before any work
        with mock.patch.object(umami._env.sys, "platform", "linux"):
            self.assertEqual(umami.device_name("gpu"), "cuda")
            with self.assertRaisesRegex(umami.UmamiError, "macOS only"):
                umami.build_umami_library("/nonexistent", None, None, {}, 1, [], device="metal")
            with self.assertRaisesRegex(umami.UmamiError, "macOS only"):
                umami.build_metal_runtime("/nonexistent")
        with mock.patch.object(umami._env.sys, "platform", "darwin"):
            self.assertEqual(umami.device_name("gpu"), "metal")


if __name__ == "__main__":
    unittest.main()
