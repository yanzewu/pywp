
import pandas as pd
import numpy as np
import sys
import re
import argparse

def analyze(filenames, flag, atol, endtol):

    result = []
    for f in filenames:
        result.append(analyze_single(f, flag, atol, endtol))
    
    if flag:
        result.sort(key=lambda x:x[2])

    df = pd.DataFrame(np.array([r[0] for r in result]), columns=result[0][1])
    if flag:
        df.insert(0, flag, np.array([r[2] for r in result]))

    df.to_csv(sys.stdout, sep='\t', index=False)


def analyze_single(filename, flag, atol=0.005, endtol=0.1):

    f = pd.read_csv(filename, '\t', index_col=False, header=0)
    # check convergence
    _match_pop = re.compile(r'[TR]\d')
    _match_momentum = re.compile(r'P[xyz][TR]\d')

    pop_indices = [c for c in f.columns if _match_pop.fullmatch(c)]
    all_indices = pop_indices + [c for c in f.columns if _match_momentum.fullmatch(c)]

    row1 = f.loc[f.shape[0]-1, pop_indices]
    row2 = f.loc[f.shape[0]-2, pop_indices]
    for j in range(len(row1)):
        if abs(row1[j] - row2[j]) > atol and not (
            row1[j] > 1-endtol and row1[j] > row2[j] and abs(row1[j]-row2[j]) < 2*atol or
            row1[j] < endtol and row1[j] < row2[j] and abs(row1[j]-row2[j]) < 2*atol):
            print('%s: Data may not be converged: (%.4f %+.4f)' % (
                filename, row1[j], row1[j]-row2[j]
            ), file=sys.stderr)
            break

    # where do we start?

    n0 = np.argmax(f.loc[0, ['P%d'%j for j in range(len(pop_indices)//2)]].values)
    flagval = f.loc[0, '%sP%d'% (flag, n0)] if flag else None

    return f.loc[f.shape[0]-2, all_indices], all_indices, flagval


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Summarize scattering probability from verbose output file (verbose=2)')

    parser.add_argument('--atol', default=0.005, type=float, help='Tolerance of population change per frame')
    parser.add_argument('--endtol', default=0.1, type=float, help='Additional criteria of convergence (>endtol or <1-endtol)')
    parser.add_argument('-f', '--flag', default='Px', help='Flag to distinguish multiple inputs. Px/Py/../Rx/Ry/..')
    parser.add_argument('filename', help='filenames', nargs='+')

    args = parser.parse_args()

    analyze(args.filename, flag=args.flag, atol=args.atol, endtol=args.endtol)


