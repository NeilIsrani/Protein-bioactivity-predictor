#pip install chembl_webresource_client
#pip install conda and rdkit
import pandas as pd
from chembl_webresource_client.new_client import new_client
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def data_creator():
    target = new_client.target
    target_query = target.search('breast cancer') #change accordingly
    targets = pd.DataFrame.from_dict(target_query)
    selected_target = targets.target_chembl_id[4] #change accordingly
    activity = new_client.activity
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res)
    df.standard_type.unique()
    df.to_csv('bioactivity_data.csv', index=False)
    df2 = df[df.standard_value.notna()]
    data_sorter(df2) 

def data_sorter(data):
    bioactivity_class = []
    for i in data.standard_value:
        if float(i) >= 10000:
            bioactivity_class.append("inactive")
        elif float(i) <= 1000:
            bioactivity_class.append("active")
        else:
            bioactivity_class.append("intermediate")
    selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
    df3 = data[selection]
    df3.to_csv('bioactivity_preprocessed_data.csv', index=False)
    df_lipinski = lipinski(df3.canonical_smiles)
    df_combined = pd.concat([df3, df_lipinski], axis=1)
    df_combined_converted = pIC50(df_combined)
    return df_combined_converted

def lipinski(smiles, verbose=False):
    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors

def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)
        
    return x
def main():
    data_creator()

if __name__ == "__main__":
    main()


