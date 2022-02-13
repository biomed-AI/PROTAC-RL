import re
import os
import argparse
import random
import pandas as pd
from rdkit import Chem
from Utils.randomized_frags import randomize_smiles, random_smi

def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-input_file", "-i",
                        required=True,
                        default='',
                        help="Original mmps file of the molecules")

    parser.add_argument("-random", "-r", required=False,
                        default=False,
                        help="Data augmentation", type=bool)

    parser.add_argument("-output_path", "-o",
                        required=True,
                        default='./test/',
                        help="Dic path of the generated src/tgt file")


    return parser.parse_args()

# input: L_2.C(C[*:2])[*:1].c1ccc([*:1])cc1.c1ccc([*:2])nc1>c1ccccc1CCc2ccccn2
# output: L_2.C(C[*])[*].c1ccc([*])cc1.c1ccc([*])nc1>c1ccccc1CCc2ccccn2
def remove_dummy_label(mmps, pattern = re.compile(r"\[.*?\]")):
    rw_mmps = []
    for mmp in mmps:
        bricks = ".".join(mmp.split(">")[0].split(".")[1:4])
        dummy_label = pattern.findall(bricks)

        for i in dummy_label:
            bricks = bricks.replace(i, "[*]")

        length = mmp.split(">")[0].split(".")[0]
        smi = mmp.split(">")[1]
        rw_mmps.append(length + "." + bricks + ">" + smi)

    return rw_mmps


def data_canonical(mmps_file, randomized_num = False):

    f = open(mmps_file, "r")
    mmps = [i.split("\n")[0] for i in f.readlines()]
    rw_mmps = remove_dummy_label(mmps)

    mmps_canonical = []

    for m in rw_mmps:
        bricks = ".".join(m.split(">")[0].split(".")[1:4])
        length = m.split(">")[0].split(".")[0]
        smi = m.split(">")[1]
        bricks = Chem.MolToSmiles(Chem.MolFromSmiles(bricks)) #

        if randomized_num:
            frags = ".".join(bricks.split(".")[1:])
            linker = bricks.split(".")[0]
            fmts = random_smi(randomized_num, frags)
            mmps_random = [length + "." + linker + "." + f + ">" + smi for f in fmts]
            mmps_random = list(set(mmps_random))
            mmps_canonical.extend(mmps_random)

        else: #
            mmps_canonical.append(length + "." + bricks + ">" + smi)

    return mmps_canonical

# data spliting(src/tgt)

def spliting(mmps):
    length = [mmp.split(".")[0] for mmp in mmps]
    groups = list(set(length))
    data = list(zip(length, mmps))
    df1 = pd.DataFrame(data, columns=["length", "mmps"])
    df2 = df1.groupby('length')

    test = []
    val = []
    train = []
    for g in groups:
        idx = list(df2.groups[g])

        l = len(idx)
        start = int(l * 0.1)
        end = int(l * 0.2)
        idx_test = idx[: start]
        idx_val = idx[start: end]
        idx_train = idx[end:]
        test = test + idx_test
        val = val + idx_val
        train = train + idx_train

    train_mmps = list(df1.iloc[train]["mmps"])
    val_mmps = list(df1.iloc[val]["mmps"])
    test_mmps = list(df1.iloc[test]["mmps"])

    return train_mmps, val_mmps, test_mmps

def write_src_tgt(mmps, name, dir):
    src_file = dir + "src" + "-" + name
    tgt_file = dir + "tgt" + "-" + name
    src = [mmp.split(".")[0] + " " + " ".join(".".join((mmp.split(">")[0]).split(".")[2:4])) for mmp in mmps]
    tgt = [" ".join(mmp.split(">")[1]) for mmp in mmps]
    with open(src_file, "w") as w:
        for j in src:
            w.write(j)
            w.write("\n")
    w.close()

    with open(tgt_file, "w") as w2:
        for j in tgt:
            w2.write(j)
            w2.write("\n")
    w2.close()



def main():
    opt = parse_args()

    mmps = data_canonical(opt.input_file, randomized_num = opt.random)
    train_mmps, val_mmps, test_mmps = spliting(mmps)
    write_src_tgt(train_mmps, name="train", dir = opt.output_path)
    write_src_tgt(val_mmps, name="val", dir = opt.output_path)
    write_src_tgt(test_mmps, name="test", dir = opt.output_path)


if __name__ == "__main__":
    main()

