"""pacrunner.uir against the goldens (pac-julia's outputs, tools/goldens/make_goldens.jl): every
program's Lowered, its rows after optimize and both encodings, value for value, on the fixture
exports (test/data/goldens) and, when it is there, the corpus (PACRUNNER_GOLDENS, else
~/symjit-data/pacrunner-goldens).

    python -m unittest discover -s test/python -p test_uir.py
"""

import dataclasses
import json
import unittest

from goldens import FIXTURE_EXPORTS, encoded_json, first_difference, golden_sets, lowered_json, op_json

from pacrunner import uir
from pacrunner.uir import LOp, Opcode


class Goldens(unittest.TestCase):
    def test_goldens(self):
        sets = golden_sets()
        self.assertTrue(any(name.startswith("goldens/") for (name, *_) in sets), "no fixture goldens in test/data/goldens")
        for (name, folder, export, manifest) in sets:
            with self.subTest(export=name):
                ls = uir.load_recurrence_export(export)
                self.assertEqual(len(ls), manifest["programs"])
                programs = json.loads((folder / "programs.json").read_text())
                for k, l in enumerate(ls):
                    self.assertEqual((l.process, l.colour), (programs[k]["process"], programs[k]["colour"]))
                    g = json.loads((folder / "lowered" / f"{k:04d}.json").read_text())
                    o = uir.optimize(l)
                    ours = {"lowered": lowered_json(l), "optimized_ops": [op_json(x) for x in o.ops],
                            "encoded": encoded_json(uir.encode(l)), "encoded_optimized": encoded_json(uir.encode(o))}
                    for key, value in ours.items():
                        diff = first_difference(g[key], value, key)
                        self.assertIsNone(diff, f"{name} program {k}: {diff}")
                    # what a device sees is the program
                    self.assertEqual(uir.decode(uir.encode(o), o).ops, o.ops)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ls = uir.load_recurrence_export(FIXTURE_EXPORTS / "pp_3j.json")

    def test_references(self):
        refs = uir.recurrence_references(FIXTURE_EXPORTS / "pp_3j.json")
        self.assertEqual(len(refs), len(self.ls))
        for l, r in zip(self.ls, refs):
            if r is not None:
                (momenta, values) = r
                self.assertEqual(len(momenta), len(values))
                self.assertTrue(all(len(v) == l.output_count() for v in values))

    def test_optimize_changes_rows_only(self):
        for l in self.ls:
            o = uir.optimize(l)
            self.assertEqual(dataclasses.replace(o, ops=l.ops), l)
            self.assertEqual(len(o.ops), len(l.ops))

    def test_validate_refuses(self):
        l = self.ls[0]
        k = next(i for i, op in enumerate(l.ops) if uir.is_vertex(op.op))
        bad = list(l.ops)
        bad[k] = dataclasses.replace(bad[k], d=len(l.edges))
        with self.assertRaises(uir.InvalidLowered):
            uir.validate(dataclasses.replace(l, ops=tuple(bad)))
        with self.assertRaises(uir.InvalidLowered):
            uir.validate(dataclasses.replace(l, ops=l.ops + (LOp(Opcode.END),)))

    def test_validate_refuses_legs(self):
        l = self.ls[0]
        crossing = tuple((k, 1.0) for k in range(l.legs))
        uir.validate(dataclasses.replace(l, crossing=crossing))
        for bad in (crossing[:-1] + ((l.legs, 1.0),), crossing[:-1] + ((-1, 1.0),), crossing[:-1] + ((0, 1.0),),
                    crossing[:-1]):
            with self.assertRaises(uir.InvalidLowered):
                uir.validate(dataclasses.replace(l, crossing=bad))
        forms = (uir.Form(((l.legs, 1.0),)),) + l.forms[1:]
        with self.assertRaises(uir.InvalidLowered):
            uir.validate(dataclasses.replace(l, forms=forms))

    def test_export_refuses_a_crossing_off_the_legs(self):
        d = uir.read_export(FIXTURE_EXPORTS / "pp_3j.json")
        p = next(p for p in d["programs"] if p["crossing"])
        p["crossing"][-1]["program_leg"] = -1
        with self.assertRaises(uir.ExportError):
            uir.load_recurrence_export(d)

    def test_wide_rows_round_trip(self):
        # a row whose indices need the second word, and one with a null operand
        rows = [LOp(Opcode.V_THREE_VECTOR, set=True, first=True, last=True, d=9000, t=2000, l=8191, r=1, x=63),
                LOp(Opcode.KERNEL, d=3, t=4, l=None, r=5, x=6),
                LOp(Opcode.REDUCE, d=1, t=2, n=70000, x=None),
                LOp(Opcode.SOURCE, d=1, t=0, n=uir.source_template(uir.PhysicalType.WEYL, True, -1, 1), l=2000)]
        words: list = []
        from pacrunner.uir.encode import _encode_row
        for op in rows:
            _encode_row(words, op)
        self.assertEqual(uir.decode_rows(words), rows)
        self.assertEqual(words[0] & 1 << 5, 1 << 5)    # EXT
        self.assertEqual(words[2] & 1 << 5, 0)         # the kernel row fits one word

    def test_encode_refuses(self):
        from pacrunner.uir.encode import _encode_row
        with self.assertRaises(uir.EncodeError):
            _encode_row([], LOp(Opcode.CLEAR, d=1 << 20, n=1))
        with self.assertRaises(uir.EncodeError):
            _encode_row([], LOp(Opcode.CLEAR, d=1, n=1, x=2))   # not one of CLEAR's fields

    def test_kernels(self):
        d = uir.read_export(FIXTURE_EXPORTS / "pp_3j.json")
        kernels = uir.export_kernels(d)
        stubs = {s.kernel for l in self.ls for s in l.stubs}
        self.assertTrue(stubs <= set(kernels.values()))
        for sha, k in kernels.items():
            self.assertEqual(k.name, "k_" + sha[:16])
        d["kernels"][0] = dict(d["kernels"][0], sha256="0" * 64)
        with self.assertRaises(uir.ExportError):
            uir.export_kernels(d)


if __name__ == "__main__":
    unittest.main()
