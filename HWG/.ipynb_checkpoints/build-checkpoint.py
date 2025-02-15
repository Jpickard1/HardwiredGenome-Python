import os
import urllib.request
import gzip
import shutil
from scipy.sparse import lil_matrix
import pandas as pd
import numpy as np
import networkx as nx
import scipy.io
import pickle


def buildHWG(thresh=600):
    print('****************')
    print('Downloading Data')
    print('****************')
    download_data()
    print('********************')
    print('Building Index Table')
    print('********************')
    indexTable = buildIndexTable()
    print('********************')
    print('Building Adj. Matrix')
    print('********************')
    HWG = buildAdjacencyMatrix(indexTable, thresh=thresh)
    print('********************')
    print('   save to a file   ')
    print('********************')
    path = load_data_path()
    fileName = 'HWG_' + str(thresh) + '.pkl'
    path = load_data_path()
    path = os.path.join(path, 'Processed', fileName)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'path={path}')
    with open(path, 'wb') as f:
        pickle.dump(HWG, f)
    return HWG

def loadHWG(thresh=600):
    fileName = f'HWG_{thresh}.pkl'
    path = load_data_path()
    path = os.path.join(path, 'Processed', fileName)
    
    if os.path.exists(path):
        print(f'Loading HWG for threshold {thresh}...')
        with open(path, 'rb') as f:
            HWG = pickle.load(f)
        print('HWG loaded successfully.')
        return HWG
    else:
        print(f'HWG for threshold {thresh} does not exist.')

        while True:
            user_input = input('Do you want to build HWG with this threshold? (yes/no): ').strip().lower()
            if user_input == 'yes':
                return buildHWG(thresh)
            elif user_input == 'no':
                new_thresh = int(input('Enter a new threshold: '))
                new_fileName = f'HWG_{new_thresh}.pkl'
                new_path = os.path.join(load_data_path(), 'Processed', new_fileName)
                if os.path.exists(new_path):
                    print(f'Loading HWG for threshold {new_thresh}...')
                    with open(new_path, 'rb') as f:
                        HWG = pickle.load(f)
                    print('HWG loaded successfully.')
                    return HWG
                else:
                    return buildHWG(new_thresh)
            else:
                print('Invalid input. Please enter "yes" or "no".')



    

### CODES TO DOWNLOAD DATA ###
def load_data_path():
    filepath = os.path.dirname(os.path.abspath(__file__))
    # Go to base path of the repo and add Data/
    data_path = os.path.join(filepath[:-4], "Data")
    return data_path
    
###This is another function I find in load_data_path.m. I feel like it might be used here(or maybe you want to put it somewhere else)
def download_data():
    """This function should call the methods to download the STRING and HURI datasets.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m
    """
    
    # Define paths
    data_path = load_data_path()
    downloaded_data_path = os.path.join(data_path, "Downloaded/")
    HuRI_path = os.path.join(downloaded_data_path, "HuRI")
    STRING_path = os.path.join(downloaded_data_path, "STRING")

    # Create directories if they don't exist
    os.makedirs(HuRI_path, exist_ok=True)
    os.makedirs(STRING_path, exist_ok=True)

    # HuRI data
    HuRI_file = os.path.join(HuRI_path, "HuRI.tsv")
    HI_union_file = os.path.join(HuRI_path, "HI-union.tsv")

    if not os.path.isfile(HuRI_file):
        urllib.request.urlretrieve("http://www.interactome-atlas.org/data/HuRI.tsv", HuRI_file)

    if not os.path.isfile(HI_union_file):
        urllib.request.urlretrieve("http://www.interactome-atlas.org/data/HI-union.tsv", HI_union_file)

    print("    HuRI Downloaded")

    # STRING data
    STRING_file = os.path.join(STRING_path, "9606.protein.links.v11.5.txt.gz")
    STRING_txt = os.path.join(STRING_path, "9606.protein.links.v11.5.txt")

    if not os.path.isfile(STRING_file):
        urllib.request.urlretrieve('https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz', STRING_file)
        with gzip.open(STRING_file, 'rb') as f_in:
            with open(STRING_txt, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("    STRING Downloaded")

    download_HumanTF()
    print("    HumanTF Downloaded")

    download_HGNC()
    print("    HGNC Downloaded")


## after running download_data(), the data is downloaded outside of current local address.I mean, it is supposed to downloaded into D:/HardwiredGenome-Python/HWG,right? but now it is placed at D:/HardwiredGenome-(or it is what it is supposed to be, I am not familiar with how matlab code work, or it is just a problme caused by name?like how it deal with "-")
def download_string():
    """This function should download the STRING database to the computer.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m lines 28-34
    """
    data_path = load_data_path()
    downloaded_data_path = os.path.join(data_path, "Downloaded/")
    STRING_path = os.path.join(downloaded_data_path, "STRING")
    # Create the directory if it doesn't exist
    os.makedirs(STRING_path, exist_ok=True)
    # Define the file paths
    STRING_file = os.path.join(STRING_path, "9606.protein.links.v11.5.txt.gz")
    STRING_txt = os.path.join(STRING_path, "9606.protein.links.v11.5.txt")
    if not os.path.isfile(STRING_file):
        urllib.request.urlretrieve('https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz', STRING_file)
        with gzip.open(STRING_file, 'rb') as f_in:
            with open(STRING_txt, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("    STRING Downloaded")
# download_string()

def download_huri():
    """This function should download the HuRI database to the computer.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m lines 18-26
    """
    # Define the path for downloaded data
    data_path = load_data_path()
    downloaded_data_path = os.path.join(data_path, "Downloaded/")
    HuRI_path = os.path.join(downloaded_data_path, "HuRI")
    
    # Create the directory if it doesn't exist
    os.makedirs(HuRI_path, exist_ok=True)
    
    # Define file paths
    HuRI_file = os.path.join(HuRI_path, "HuRI.tsv")
    HI_union_file = os.path.join(HuRI_path, "HI-union.tsv")
    
    # Download the files if they don't exist
    if not os.path.isfile(HuRI_file):
        urllib.request.urlretrieve("http://www.interactome-atlas.org/data/HuRI.tsv", HuRI_file)
    
    if not os.path.isfile(HI_union_file):
        urllib.request.urlretrieve("http://www.interactome-atlas.org/data/HI-union.tsv", HI_union_file)
    
    print("    HuRI Downloaded")
    


def download_HGNC():
    """This function should download the HGNC database to the computer.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m lines 46-54
    """
    data_path = load_data_path()
    downloaded_data_path = os.path.join(data_path, "Downloaded/")
    HGNC_path = os.path.join(downloaded_data_path, "HGNC")
    # Create the directory if it doesn't exist
    os.makedirs(HGNC_path, exist_ok=True)
    # Define the file path
    HGNC_file = os.path.join(HGNC_path, "gene_lookup_dictionary.tsv")
    if not os.path.isfile(HGNC_file):
        urllib.request.urlretrieve('https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt', HGNC_file)


def download_HumanTF():
    """This function should download the Human Transcription Factors database to the computer.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m lines 56-64
    """
    data_path = load_data_path()
    downloaded_data_path = os.path.join(data_path, "Downloaded/")
    HuTF_path = os.path.join(downloaded_data_path, "HuTF")
    # Create the directory if it doesn't exist
    os.makedirs(HuTF_path, exist_ok=True)
    HuTF_file = os.path.join(HuTF_path, "HuTF_db.csv")
    if not os.path.isfile(HuTF_file):
        urllib.request.urlretrieve('http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv', HuTF_file)

### extra function needed for buildIndexTable()
def stable_ID_map(oldIDs, to_gene, uni):##extra needed function, converted from stable_ID_map.m
        # Set data path
    data_path = load_data_path()

    # Read table
    GRCh38_data_path = os.path.join(data_path, "Downloaded/Ensemble/GRCh38.p13.tsv")
    GRCh = pd.read_csv(GRCh38_data_path, sep='\t')
    biomart_custom_data_path = os.path.join(data_path, "Downloaded/Ensemble/biomart_stable_gene_protein_id_map.txt")
    biomart_custom = pd.read_csv(biomart_custom_data_path, sep='\t')
    STRING_ID_map_path = os.path.join(data_path, "Downloaded/STRING/string_9606_ENSG_ENSP_10_all_T.tsv")
    STRING_IDs = pd.read_csv(STRING_ID_map_path, sep='\t', header=None)
    STRING_IDs.columns = ['0', 'GENE ID', 'PROTEIN ID']

    # Extract stable IDs
    gene_IDs_GRCh = GRCh['Gene stable ID'].str.strip()
    protein_IDs_GRCh = GRCh['Protein stable ID'].str.strip()
    gene_IDs_biomart = biomart_custom['Gene stable ID'].str.strip()
    protein_IDs_biomart = biomart_custom['Protein stable ID'].str.strip()
    gene_IDs_STRING = STRING_IDs['GENE ID'].str.strip()
    protein_IDs_STRING = STRING_IDs['PROTEIN ID'].str.strip()

    # Stack the lists of IDs from multiple sources
    protein_IDs = pd.concat([protein_IDs_STRING, protein_IDs_GRCh, protein_IDs_biomart]).reset_index(drop=True)
    gene_IDs = pd.concat([gene_IDs_STRING, gene_IDs_GRCh, gene_IDs_biomart]).reset_index(drop=True)

    if uni:
        return gene_IDs, protein_IDs

    # Reduce the map
    PIDs = protein_IDs.astype(str)
    GIDs = gene_IDs.astype(str)
    P_missing = PIDs[PIDs == ''].index
    G_missing = GIDs[GIDs == ''].index
    missing = np.unique(np.concatenate([P_missing, G_missing]))
    keep = np.ones(len(protein_IDs), dtype=bool)
    keep[missing] = False
    protein_IDs = protein_IDs[keep]
    gene_IDs = gene_IDs[keep]

    # Build a map between the genes
    if to_gene:
        id_map = dict(zip(protein_IDs, gene_IDs))
    else:
        id_map = dict(zip(gene_IDs, protein_IDs))

    # Build newIDs
    newIDs = []
    unmapped = []
    for i, old_id in enumerate(oldIDs):
        #print(old_id)
        new_id = id_map.get(old_id, "")
        newIDs.append(new_id)
        if new_id == "":
            unmapped.append(i)

    return newIDs, unmapped


### CODES TO BUILD INDEX TABLE ###
def buildIndexTable():
    """This function should build an index table to map each entry of the A matrix to gene names.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/buildIndexTable.m
    """
    # Path to find downloaded data
    data_path = load_data_path()
    
    # Get unique genes used in the HWG
    gene_ids, _ = stable_ID_map(0, 0, True)
    gene_ids = pd.unique(gene_ids)
    
    # Create index table
    indexTable = pd.DataFrame({
        'Stable ID': gene_ids,
        'Transcription Factor': [False] * len(gene_ids)
    })
    
    # Fill in transcription factors
    HuTF_downloaded_data =os.path.join(data_path,"Downloaded/HuTF/HuTF_db.csv")
    HuTF_db = pd.read_csv(HuTF_downloaded_data)
    isTF = HuTF_db['Is TF?']
    gene_stable_ids = HuTF_db['Ensembl ID']
    tf_ids = gene_stable_ids[isTF == "Yes"]
    for id in tf_ids:
        idx = indexTable[indexTable['Stable ID'] == id].index
        if idx.empty:
            print(f"    Missing TF: {id}")
        else:
            indexTable.loc[idx, 'Transcription Factor'] = True
            
    # Add in gene names where known
    # ENSEMBLE DATA
    GRCh_downloaded_data =os.path.join(data_path, "Downloaded/Ensemble/GRCh38.p13.tsv")
    GRCh_genes = pd.read_csv(GRCh_downloaded_data, sep='\t')
    GRCh_genes['Gene stable ID'] = GRCh_genes['Gene stable ID'].str.strip()
    GRCh_genes['Gene name'] = GRCh_genes['Gene name'].str.strip()
    
    # HGNC DATA
    HGNC_downloaded_data = os.path.join(data_path, "Downloaded/HGNC/gene_lookup_dictionary.tsv")
    HGNC_genes = pd.read_csv(HGNC_downloaded_data, sep='\t')
    HGNC_genes['ensembl_gene_id'] = HGNC_genes['ensembl_gene_id'].str.strip()
    HGNC_genes['symbol'] = HGNC_genes['symbol'].str.strip()
    
    # Merge in GRCH
    indexTable = pd.merge(indexTable, GRCh_genes, how='left',
                   left_on='Stable ID',
                   right_on='Gene stable ID',
                  )
    indexTable = indexTable[indexTable['Stable ID'].notna()]
    indexTable = indexTable.drop_duplicates(subset=['Stable ID'], keep='first')
    indexTable = indexTable.reset_index()
    indexTable = indexTable.drop('index', axis=1)
    
    # Merge in HGNC
    indexTable = pd.merge(indexTable, HGNC_genes, how='left',
                   left_on='Stable ID',
                   right_on='ensembl_gene_id',
                  )
    indexTable = indexTable[indexTable['Stable ID'].notna()]
    indexTable = indexTable.drop_duplicates(subset=['Stable ID'], keep='first')
    indexTable = indexTable.reset_index()
    indexTable = indexTable.drop('index', axis=1)
    
    # Reduce the table
    columns2keep = ['Stable ID', 'symbol', 'Transcription Factor', 'Gene start (bp)', 'Gene end (bp)', 'Chromosome/scaffold name', 'Protein stable ID']
    indexTable = indexTable[columns2keep]

    return indexTable

##extra function needed for buildAdjacencyMatrix()
def list_HuRI():
    data_path = load_data_path()
    HuRI_downloaded_data = os.path.join(data_path, "Downloaded/HuRI/HI-union.tsv")

    HuRI_PPI = pd.read_csv(HuRI_downloaded_data, sep='\t', header=None)

    # Create the DataFrame
    A_list = pd.DataFrame({
        'Gene 1': HuRI_PPI[0].values, #.astype(str),
        'Gene 2': HuRI_PPI[1].values, #['Protein_2'].astype(str)
    })
    return A_list

def list_STRING(thresh, STRING_PPI=None):
    print("Accessing STRING data")

    # Set data path
    data_path = load_data_path()

    # Read and sort raw PPIs
    if STRING_PPI is None or not isinstance(STRING_PPI, pd.DataFrame):
        print("    Reading data from file...")
        STRING_downloaded_PPI = f"{data_path}/Downloaded/STRING/9606.protein.links.v11.5.txt"
        STRING_PPI = pd.read_csv(STRING_downloaded_PPI, delim_whitespace=True)
        print("    File read complete")
    else:
        print("    STRING data present from previous file read")

    # Apply threshold
    confident_indices = STRING_PPI['combined_score'].astype(int) > thresh
    P1_confident = STRING_PPI['protein1'][confident_indices].astype(str)
    P2_confident = STRING_PPI['protein2'][confident_indices].astype(str)

    # Calculate percent removed
    rem = 1 - len(P1_confident) / len(STRING_PPI)
    print(f"    {rem * 100:.2f}% of the data was removed because it is below the confidence threshold.")

    # Convert Protein IDs to Gene IDs
    # Remove species identifier 9606
    P1_confident = P1_confident.str[5:]
    P2_confident = P2_confident.str[5:]

    # Map protein IDs to gene IDs
    P1_gene_id, remove1 = stable_ID_map(P1_confident.tolist(), True, uni=False)
    P2_gene_id, remove2 = stable_ID_map(P2_confident.tolist(), True, uni=False)

    # Remove rows that were unmapped
    remove_rows = np.unique(np.concatenate([remove1, remove2]))
    idxs = np.ones(len(P1_gene_id), dtype=bool)
    idxs[remove_rows] = False
    P1_gene_id = np.array(P1_gene_id)[idxs]
    P2_gene_id = np.array(P2_gene_id)[idxs]

    # Calculate further removal statistics
    rem = 1 - len(P1_gene_id) / len(P1_confident)
    print(f"    {rem * 100:.2f}% of the data was removed because it wasn't mapped to a stable gene ID.")

    # Organize data as a DataFrame
    A_list = pd.DataFrame({
        'Gene 1': P1_gene_id,
        'Gene 2': P2_gene_id
    })

    # Calculate total usage
    rem = len(P1_gene_id) / len(STRING_PPI)
    print(f"STRING Data Access Complete: {rem * 100:.2f}% of the data was used.")
    return A_list, STRING_PPI

def list_combine(A_lists):
    A_list = pd.DataFrame(columns=['Gene 1', 'Gene 2'])
    
    # Stack the adjacency lists from each database
    for adj_list in A_lists:
        A_list = pd.concat([A_list, adj_list], ignore_index=True)
    
    return A_list

def list2mat(A_list, indexTable):
    unique_ids = indexTable['Stable ID']
    n = len(unique_ids)
    gene_idxs = {id_: idx for idx, id_ in enumerate(unique_ids)}

    # Construct a matrix
    A = lil_matrix((n, n), dtype=int)
    for index, row in A_list.iterrows():
        protein1 = row['Gene 1']
        protein2 = row['Gene 2']
        if protein1 in gene_idxs and protein2 in gene_idxs:
            A[gene_idxs[protein1], gene_idxs[protein2]] = 1

    # Symmetrize the matrix
    A = A + A.T
    A[A > 1] = 1  # Ensuring that the matrix contains only 1s and 0s after symmetrization

    return A

def reduceHWG(A, indexTable):
    # Reduce to only include interacting genes
    interacting_genes = np.where(np.sum(A, axis=0) > 0)[0]
    A_reduced = A[interacting_genes][:, interacting_genes]
    indexTable_reduced = indexTable.iloc[interacting_genes]

    # Reduce to the largest connected component
    G = nx.from_numpy_array(A_reduced)
    largest_component = max(nx.connected_components(G), key=len)
    keep = list(largest_component)

    A_HWG = A_reduced[np.ix_(keep, keep)]
    nodeTable = indexTable_reduced.iloc[keep]

    return A_HWG, nodeTable

def HWGMatrixDecomp(A_HWG, A_IndexTable):
    TF_idxs = A_IndexTable['Transcription Factor'].values
    
    # B matrix (indexed using both the A and C tables)
    B_HWG = A_HWG[:, TF_idxs]
    
    # C matrix
    C_HWG = A_HWG[TF_idxs][:, TF_idxs]
    C_IndexTable = A_IndexTable[TF_idxs]

    return B_HWG, C_HWG, C_IndexTable

### CODES TO BUILD HARDWIRED GENOME A MATRIX ###
def buildAdjacencyMatrix(indexTable, thresh = 600):
    """This function assembles the STRING and HuRI downloaded data into an adjacency matrix.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/buildHWGobj.m
    """

    # Get adjacency lists
    A_list_HuRI = list_HuRI()
    print("HuRI Data Accessed")

    A_list_STRING, _ = list_STRING(thresh)
    print("STRING Data Accessed")

    # Combine the lists
    A_list = list_combine([A_list_HuRI, A_list_STRING])
    
    # Build the matrix
    A_HWG = list2mat(A_list, indexTable)

    # Reduce the matrix and list to only contain used genes
    # A_HWG, A_IndexTable = reduceHWG(A_HWG, indexTable)

    # Construct the factor matrices
    # B_HWG, C_HWG, C_IndexTable = HWGMatrixDecomp(A_HWG, A_IndexTable)

    # Package as the Hardwired Genome object
    HWG = {
        'thresh': thresh,
        'A': A_HWG,
        # 'B': B_HWG,
        # 'C': C_HWG,
        'geneIndexTable': indexTable,
        # 'TFIndexTable': C_IndexTable
    }

    return HWG
