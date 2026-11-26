"""The hand-compiled SIMD fragments (pacrunner/simd/handgen.py, lib_hand_<isa>/): up to date with handgen.py, and
lib/'s bits: each fragment run by a small MIR interpreter (Python floats: IEEE doubles, the same operations
one lane does) on random inputs equals lib/'s fragment bit for bit; the registers stay in the ISA's pool."""
import json
import random
import struct
import unittest

from pacrunner.simd import handgen, library as L

OPS = {"Plus": lambda a, b: a + b, "Minus": lambda a, b: a - b, "Times": lambda a, b: a * b,
       "Divide": lambda a, b: a / b}


def const(c):
    return c if isinstance(c, float) else struct.unpack("<d", struct.pack("<Q", int(c, 16)))[0]


def run(frag, xs):
    """the outputs of a fragment's MIR on the inputs xs (one lane)"""
    mem = list(xs) + [0.0] * frag["outputs"]
    stack, regs = {}, {}

    def key(r):
        return r if isinstance(r, str) else tuple(r.items())[0]

    def load(loc):
        ((k, v),) = loc.items()
        return mem[v] if k == "Mem" else stack[v]

    for ins in frag["code"]:
        if isinstance(ins, str):
            continue
        ((k, v),) = ins.items()
        if k == "Load":
            regs[key(v["dst"])] = load(v["loc"])
        elif k == "LoadMath":
            regs[key(v["dst"])] = OPS[v["op"]](regs[key(v["s1"])], load(v["loc"]))
        elif k == "Bi":
            regs[key(v["dst"])] = OPS[v["op"]](regs[key(v["s1"])], regs[key(v["s2"])])
        elif k == "Uni" and v["op"] == "Neg":
            regs[key(v["dst"])] = -regs[key(v["s1"])]
        elif k == "LoadConst":
            regs[key(v["dst"])] = const(frag["consts"][v["idx"]])
        elif k == "Save":
            ((lk, lv),) = v["loc"].items()
            (mem if lk == "Mem" else stack)[lv] = regs[key(v["src"])]
        else:
            raise ValueError(f"{frag['name']}: {k}")
    return mem[len(xs):]


def bits(x):
    return struct.pack("<d", x)


def registers(frag):
    out = set()
    for ins in frag["code"]:
        if isinstance(ins, dict):
            ((_, v),) = ins.items()
            out |= {r["Gen"] for r in v.values() if isinstance(r, dict) and "Gen" in r}
    return out


class TestHandgen(unittest.TestCase):
    def test_up_to_date(self):
        for isa in handgen.POOLS:
            for name in filter(L.hand_compiled, L.catalogue()):
                with self.subTest(isa=isa, fragment=name):
                    shipped = json.loads((L.hand_dir(isa) / L.file_of(name).name).read_text())
                    self.assertEqual(shipped, json.loads(json.dumps(handgen.hand_fragment(name, isa))),
                                     "regenerate: python -m pacrunner.simd.handgen")

    def test_bits_of_lib(self):
        rng = random.Random(1)
        for isa in handgen.POOLS:
            for name in filter(L.hand_compiled, L.catalogue()):
                with self.subTest(isa=isa, fragment=name):
                    lib, hand = L.fragment(name), L.fragment(name, isa)
                    self.assertEqual(hand["inputs"], lib["inputs"])
                    self.assertEqual(hand["outputs"], lib["outputs"])
                    self.assertLessEqual(registers(hand), set(handgen.POOLS[isa]))
                    for _ in range(100):
                        xs = [rng.uniform(-2.0, 2.0) for _ in lib["inputs"]]
                        self.assertEqual([bits(y) for y in run(hand, xs)], [bits(y) for y in run(lib, xs)])

    def test_kernels_take_their_isa(self):
        name = "vertex:weyl_vector_a:set"
        self.assertIn("hand", L.fragment(name, "avx512"))
        self.assertNotIn("hand", L.fragment(name))
        self.assertNotIn("hand", L.fragment("wave:gluon:1", "avx512"))


if __name__ == "__main__":
    unittest.main()
