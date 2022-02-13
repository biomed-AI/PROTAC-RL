
# import warnings
# warnings.filterwarnings("ignore")
import os
import argparse
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMMPA
from rdkit.Chem import Lipinski, Descriptors
from joblib import Parallel, delayed
from multiprocessing import Pool

def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-input_data_path", "-i", required=True,
                        help="SDF file of the molecules")

    parser.add_argument("-n_jobs", "-n",
                         required=True,
                         default=1,
                         help="number of the cores", type=int)

    parser.add_argument("-output", "-o", required=True,
                        help="Generated mmps pairs")


    return parser.parse_args()


def filter(mol, type = "frags"):

    HBD = Lipinski.NumHDonors(mol)
    HBA = Lipinski.NumHAcceptors(mol)
    rings = len(Chem.GetSymmSSSR(mol))
    MW = Chem.Descriptors.MolWt(mol)

    if type == "frags":
        action = (HBD <=8) & (HBA <=8) & (rings >= 1) & (MW <=800)
    else:
        action = (HBD <= 5) & (HBA <= 5) & (MW <= 500)

    return action


# MMPs cutting algorithm
def mmps_cutting(mol, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]", dummy=True, filtering=True):
    """ MMPs function"""
    FMQs = []
    fmq = None
    #mol = Chem.MolFromSmiles(smi)
    try:
        smi = Chem.MolToSmiles(mol)
        bricks = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, \
                                    pattern=pattern, resultsAsMols=False)

        for linker, chains in bricks:

            linker_mol = Chem.MolFromSmiles(linker)
            linker_size = linker_mol.GetNumHeavyAtoms()
            linker_site_idxs = [atom.GetIdx() for atom in linker_mol.GetAtoms() if atom.GetAtomicNum() == 0]
            linker_length = len(Chem.rdmolops.GetShortestPath(linker_mol, \
                                                              linker_site_idxs[0], linker_site_idxs[1])) - 2

            if (linker_size >= 2) & (linker_length >= 1):
                frag1_mol = Chem.MolFromSmiles(chains.split(".")[0])
                frag2_mol = Chem.MolFromSmiles(chains.split(".")[1])
                frag1_size = frag1_mol.GetNumHeavyAtoms()
                frag2_size = frag2_mol.GetNumHeavyAtoms()


                if (frag1_size >= 5) & ((frag2_size >= 5) & ((frag1_size + frag1_size) >= linker_size)):

                    if filtering:

                        action = filter(linker_mol, type="frags") & filter(frag1_mol, type="frags") \
                                 & filter(frag2_mol, type="frags")
                        if action:

                            if dummy:
                                fmq = "L_" + str(linker_length) + "." + "%s" % (linker) + "." \
                                      + "%s" % (chains) + ">" + "%s" % (smi)
                            else:
                                fmq = "L_" + str(linker_length) + "." + "%s" % (linker) + "." \
                                      + "%s" % (remove_dummys(chains)) + ">" + "%s" % (smi)
                    else:

                        if dummy:
                            fmq = "L_" + str(linker_length) + "." + "%s" % (linker) + "." \
                                  + "%s" % (chains) + ">" + "%s" % (smi)
                        else:
                            fmq = "L_" + str(linker_length) + "." + "%s" % (linker) + "." \
                                  + "%s" % (remove_dummys(chains)) + ">" + "%s" % (smi)

                    FMQs.append(fmq)
    except:
        print("error")
        FMQs = []

    return FMQs


# remove dummy atoms(*) from MOL/SMILES format
def remove_dummys(smi_string):
    return Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string), \
                                                                    Chem.MolFromSmiles('*'), \
                                                                    Chem.MolFromSmiles('[H]'), True)[0]))


def remove_dummys_mol(smi_string):
    return Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string), \
                                                   Chem.MolFromSmiles('*'), \
                                                   Chem.MolFromSmiles('[H]'), True)[0])


# fmq (dummy) example: L_2.C(C[*:2])[*:1].c1ccc([*:1])cc1.c1ccc([*:2])nc1 >c1ccccc1CCc2ccccn2

def main():
    opt = parse_args()
    mols = Chem.SDMolSupplier(opt.input_data_path)
    fmqs = Parallel(n_jobs=opt.n_jobs)(delayed(mmps_cutting)(i) for i in mols)

    fmqs = [j for i in fmqs for j in i if j]
    fmqs = list(set(fmqs))
    #
    w = open(opt.output, "w")
    for fmq in fmqs:
        w.write(fmq)
        w.write("\n")
    w.close()

if __name__ == "__main__":
    main()



