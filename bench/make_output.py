"""make_output.py OUTPUT PROCESS...: the MadGraph7 output of the processes (the first `generate`, the others `add process`)
with the run card of MadGraph7's test_mg7_amplicol (fixed scales, NNPDF23, seed 7). MadGraph7: MG7 or
PACRUNNER_MADGRAPH7. Run with MadGraph7's python."""
import os
import sys

mg = os.environ.get("MG7") or os.environ.get("PACRUNNER_MADGRAPH7")
if not mg:
    sys.exit("MG7 or PACRUNNER_MADGRAPH7: the MadGraph7 checkout")
sys.path.insert(0, mg)
sys.path.insert(0, os.path.join(mg, "tests", "acceptance_tests"))
import madgraph.interface.master_interface as MGCmd  # noqa: E402
from test_mg7_amplicol import _edit_run_card  # noqa: E402

out, *procs = sys.argv[1:]
cmd = MGCmd.MasterCmd()
cmd.no_notification()
cmd.exec_cmd("set automatic_html_opening False --no_save")
for i, p in enumerate(procs):
    cmd.exec_cmd(("generate " if i == 0 else "add process ") + p)
cmd.exec_cmd("output mg7 %s" % out)
_edit_run_card(os.path.join(out, "Cards", "run_card.toml"))
