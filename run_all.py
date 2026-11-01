"""
RUN ALL · the "on" button
=========================

IN ONE LINE
    Runs steps 01-06 and 09 in order and counts how many numbers match the paper.

WHAT HAPPENS
    1. Runs each step as a separate program, one after the other.
    2. Stops at the first step that crashes and shows the error.
    3. Counts the lines "check ... same" / "check ... DIFFERENT" printed by the steps.

INPUT    data/Dados Rebio Jaru 2006 a 2008.xlsx
OUTPUT   everything in work/ and results/
TIME     5-15 min
RUN      python run_all.py            (python run_all.py 03 04  runs only those steps)
SUCCESS  the last line reads  "Done. Checks: 116 same, 0 different."
"""
import subprocess
import sys

STEPS = ['01_prepare', '02_select_terms', '03_final_closure', '04_checks_2008', '05_validate_years',
         '06_explore', '09_figures']
chosen = [s for s in STEPS if not sys.argv[1:] or s[:2] in sys.argv[1:]]
total = {'same': 0, 'DIFFERENT': 0}
for step in chosen:
    print(f'\n=== {step}', flush=True)
    out = subprocess.run([sys.executable, f'{step}.py'], capture_output=True, text=True)
    print(out.stdout[-3000:], out.stderr[-2000:] if out.returncode else '', flush=True)
    if out.returncode:
        sys.exit(f'{step} stopped with an error (above).')
    for k in total:
        total[k] += sum(line.strip().startswith('check') and line.rstrip().endswith(k) for line in out.stdout.splitlines())
print(f"\nDone. Checks: {total['same']} same, {total['DIFFERENT']} different.")
