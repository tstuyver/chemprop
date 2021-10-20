import pandas as pd
import numpy as np
import rdkit.Chem as Chem

df = pd.read_csv("sn2.csv")

df2 = pd.read_pickle("scaled_desc.pkl")

reactant_list = df["rxn_smiles"].tolist()

ordered_desc = []


def rbf_expansion(expanded, mu=0, delta=0.01, kmax=8):
    k = np.arange(0, kmax)
    return np.exp(-(expanded - (mu + delta * k))**2 / delta)


for reactant in reactant_list:
    mol = Chem.MolFromSmiles(reactant)
    print([atom.GetSymbol() for atom in mol.GetAtoms()])
    charge = []
    fukui_neu = []
    fukui_elec = []
    nmr = []

    for smiles in reactant.split("."):
        for i in range(len([atom.GetSymbol() for atom in Chem.MolFromSmiles(smiles).GetAtoms()])):
            charge.append([df2.loc[smiles, "partial_charge"][i]])
            fukui_elec.append([df2.loc[smiles, "fukui_elec"][i]])
            fukui_neu.append([df2.loc[smiles, "fukui_neu"][i]])
            nmr.append([df2.loc[smiles, "NMR"][i]])

    charge = np.apply_along_axis(rbf_expansion, -1, charge, -2.0, 0.06, 50)
    fukui_elec = np.apply_along_axis(rbf_expansion, -1, fukui_elec, 0, 0.02, 50)
    fukui_neu = np.apply_along_axis(rbf_expansion, -1, fukui_neu, 0, 0.02, 50)
    nmr = np.apply_along_axis(rbf_expansion, -1, nmr, 0.0, 0.06, 50)
    ordered_desc.append([reactant, np.array(charge), np.array(fukui_elec), np.array(fukui_neu), np.array(nmr)])
    #ordered_desc.append([reactant,np.array(nmr)])

ordered_desc_df = pd.DataFrame(ordered_desc, columns=["smiles","charge","fukui_elec","fukui_neu","nmr"])
#ordered_desc_df = pd.DataFrame(ordered_desc, columns=["smiles","nmr"])

ordered_desc_df = ordered_desc_df.set_index("smiles")

print(ordered_desc_df.head())
ordered_desc_df.to_pickle("ordered_desc_sn2.pkl")
