import os
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers,rdMolAlign
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig

# Set up features to use in FeatureMap
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
fdef = AllChem.BuildFeatureFactory(fdefName)

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable',
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')


def get_FeatureMapScore(query_mol, ref_mol):
    featLists = []
    for m in [query_mol, ref_mol]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = FeatMaps.FeatMapScoreMode.Best
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))

    return fm_score


def calc_SC_RDKit_score(query_mol, ref_mol):
    fm_score = get_FeatureMapScore(query_mol, ref_mol)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
                                                     allowReordering=False)
    SC_RDKit_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)

    return SC_RDKit_score

def get_frags_mol_from_gen(mol, frags_smi):
    F = []
    "frags_smi = H replace dummy atoms"
    #Chem.Kekulize(mol, clearAromaticFlags=True)
    frags_mol = Chem.MolFromSmiles(frags_smi)
    #Chem.Kekulize(frags_mol, clearAromaticFlags=True)
    matches = list(mol.GetSubstructMatches(frags_mol))

    atoms = mol.GetNumAtoms()
    atoms_list = list(range(atoms))

    for match in matches:
        linker_list = list(set(atoms_list).difference(set(match)))
        linker_list = sorted(linker_list, reverse=True)

        mol_rw = Chem.RWMol(mol)
        for idx in linker_list:
            mol_rw.RemoveAtom(idx)
        frags = Chem.Mol(mol_rw)
        if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
            F.append(frags)
    return F

def uncharge(mol):

    rdMolStandardize.ChargeParent(mol)

    return mol

# only one frags
def get_frags_mol_from_ref(mol, frags_smi, linker_smi):
    match = []
    #Chem.Kekulize(mol, clearAromaticFlags=True)
    frags_mol = Chem.MolFromSmiles(frags_smi)
    linker_mol = Chem.MolFromSmiles(linker_smi)
    #Chem.Kekulize(frags_mol, clearAromaticFlags=True)
    #Chem.Kekulize(linker_mol, clearAromaticFlags=True)
    matches_frags = list(mol.GetSubstructMatches(frags_mol))
    matches_linker = list(mol.GetSubstructMatches(linker_mol))

    atoms = mol.GetNumAtoms()
    for i in matches_frags:
        for j in matches_linker:
            if len(list(set(i + j))) == atoms:
                match = i
                break

    atoms_list = list(range(atoms))

    linker_list = list(set(atoms_list).difference(set(match)))
    linker_list = sorted(linker_list, reverse=True)

    mol_rw = Chem.RWMol(mol)
    for idx in linker_list:
        mol_rw.RemoveAtom(idx)
    frags = Chem.Mol(mol_rw)
    # if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
    return frags


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-fragments", "-f", required=True,
                        help="Fragments smiles of the original molecules")

    parser.add_argument("-linker", "-l", required=True,
                        help="Linker smiles of the original molecules")# replace dummy atom using H)

    parser.add_argument("-reference", "-r", required=True,
                        help="Reference 3D conformation (SDF format)")

    parser.add_argument("-input_mol", "-i", required=True,
                        help="Generated molecules (SDF format)")

    parser.add_argument("-output", "-o", required=True,
                        help="Generated molecules with the rmsd and the SC similarity value(SDF format)")

    opt = parser.parse_args()
    frags = opt.fragments
    linker = opt.linker

    ref_mol = Chem.SDMolSupplier(opt.reference, removeHs = False)[0]
    gen_mol = Chem.SDMolSupplier(opt.input_mol, removeHs = False)

    # similarity
    similarity = []
    ref = Chem.RemoveHs(ref_mol)
    for i in range(len(gen_mol)):
        g= Chem.RemoveHs(gen_mol[i])
        #Chem.SanitizeMol(ref_mol)
        #Chem.SanitizeMol(g_mol)
        # Align
        try:
            pyO3A = rdMolAlign.GetO3A(g, ref).Align()
            similarity.append(calc_SC_RDKit_score(g, ref))
        except:
            similarity.append("none" + str(i))

    # rmsd
    frags_rmsd = []
    #ref = Chem.RemoveHs(ref_mol)
    frags_ref = get_frags_mol_from_ref(Chem.RemoveHs(ref_mol), frags, linker)
    Chem.SanitizeMol(frags_ref)

    for i in range(len(gen_mol)):
        try:
            frags_gen = get_frags_mol_from_gen(Chem.RemoveHs(gen_mol[i]), frags)

            Chem.SanitizeMol(frags_gen[0])
            rms_mini = rdMolAlign.GetBestRMS(frags_gen[0], frags_ref)
            length = len(frags_gen)

            if length > 1:
                for j in range(1, length):
                    Chem.SanitizeMol(frags_gen[j])  # pass kekulize
                    c = rdMolStandardize.Uncharger()  # # pass charge
                    uncharge_mol = c.uncharge(frags_gen[j])
                    rms = rdMolAlign.GetBestRMS(uncharge_mol, frags_ref)  # find the mini rms
                    if rms < rms_mini:
                        rms_mini = rms
        except:
            rms_mini = "none"
        frags_rmsd.append(rms_mini)

    # write into sdf file
    w = Chem.SDWriter(opt.output)
    for i in range(len(gen_mol)):
        m = gen_mol[i]
        #AllChem.Compute2DCoords(m)
        m.SetProp("RMSD", str(frags_rmsd[i]))
        m.SetProp("Smilarity", str(similarity[i]))
        w.write(m)
    w.close()


if __name__ == "__main__":
    main()

