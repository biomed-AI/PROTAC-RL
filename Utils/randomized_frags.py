
import argparse
import rdkit
import random
from rdkit import Chem
# from onmt.utils.logging import logger

def randomize_smiles(mol, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    if not mol:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    if random_type == "restricted":
        # new_atom_order = list(range(mol.GetNumHeavyAtoms()))
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    raise ValueError("Type '{}' is not valid".format(random_type))

def save(smis, file, length=0):
    "save SMILES into a txt file \
     src format \
     length : the shortest linker atom (list) defined by user"

    f = open(file, "w")
    if length == 0:
        # logger.info("Missing SLBD information!")
        print("Missing SLBD information!")
    else:
        for s in smis:
            W = ["L_" + str(l) + " " + " ".join(s) for l in length.split(",") if str(l) != "0"]
            for w in W:
                f.write(w)
                f.write("\n")
    f.close()

    return f

def random_smi(num, smi):
    "given a SMILES (fragments SMILES) \
     return a list of ramdomized SMILES (number equal to [num])"

    fmts = []
    mol = Chem.MolFromSmiles(smi)
    fmts.append(smi)
    for i in range(50000):
        new = randomize_smiles(mol, random_type="restricted")
        fmts.append(new)
        fmts = list(set(fmts))
        if len(fmts) == num:
            break
        else:
            continue

    return fmts

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-num", "-n", required=True,
                        help="Size of the randomized SMILES List")
    parser.add_argument("-smi", "-s", required=True,
                        help="SMILES of fragments")
    parser.add_argument("-length", "-l", required=True,
                        help="List of SLBD")
    parser.add_argument("-output", "-o", required=True,
                        help="Output file")
    opt = parser.parse_args()
    fmts = random_smi(opt.num, opt.smi)
    save(fmts, opt.output, opt.length)


if __name__ == "__main__":
    main()