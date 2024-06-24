import os
import urllib.request
import gzip
import shutil
### CODES TO DOWNLOAD DATA ###
def load_data_path():
    filepath = os.path.dirname(os.path.abspath(__file__))
    # Go to base path of the repo and add Data/
    data_path = os.path.join(filepath[:-4], "Data/")
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
##print("Current Working Directory:", os.getcwd())


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
        urllib.request.urlretrieve('http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt', HGNC_file)

    print("    HGNC Downloaded")


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

    print("    HuTF Downloaded")

### extra function needed for buildIndexTable()
def stable_ID_map(oldIDs, to_gene, uni):##extra needed function, converted from stable_ID_map.m
        # Set data path
    data_path = load_data_path()

    # Read table
    GRCh38_data_path = data_path + "Downloaded/Ensemble/GRCh38.p13.tsv"
    GRCh = pd.read_csv(GRCh38_data_path, sep='\t')
    biomart_custom_data_path = data_path + "Downloaded/Ensemble/biomart_stable_gene_protein_id_map.txt"
    biomart_custom = pd.read_csv(biomart_custom_data_path, sep='\t')
    STRING_ID_map_path = data_path + "Downloaded/STRING/string_9606_ENSG_ENSP_10_all_T.tsv"
    STRING_IDs = pd.read_csv(STRING_ID_map_path, sep='\t')

    # Extract stable IDs
    gene_IDs_GRCh = GRCh['Gene_stable_ID'].str.strip()
    protein_IDs_GRCh = GRCh['Protein_stable_ID'].str.strip()
    gene_IDs_biomart = biomart_custom['Gene_stable_ID'].str.strip()
    protein_IDs_biomart = biomart_custom['Protein_stable_ID'].str.strip()
    gene_IDs_STRING = STRING_IDs['GENE_ID'].str.strip()
    protein_IDs_STRING = STRING_IDs['PROTEIN_ID'].str.strip()

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
    data_path = load_data_path()

    # Get unique genes
    gene_ids, _ = stable_ID_map(0, 0, True)
    gene_ids = pd.unique(gene_ids)

    # Create index table
    indexTable = pd.DataFrame({
        'Stable ID': gene_ids,
        'Gene Name': [None] * len(gene_ids),
        'Transcription Factor': [False] * len(gene_ids)
    })

    # Add in Transcription Factors
    HuTF_downloaded_data = data_path + "Downloaded/HuTF/HuTF_db.csv"
    HuTF_db = pd.read_csv(HuTF_downloaded_data)

    isTF = HuTF_db['IsTF_']
    gene_stable_ids = HuTF_db['EnsemblID']

    tf_ids = gene_stable_ids[isTF == "Yes"]
    for id in tf_ids:
        idx = indexTable[indexTable['Stable ID'] == id].index
        if idx.empty:
            print(f"    Missing TF: {id}")
        else:
            indexTable.loc[idx, 'Transcription Factor'] = True

    # Add in gene names where known
    # ENSEMBLE DATA
    GRCh_downloaded_data = data_path + "Downloaded/Ensemble/GRCh38.p13.tsv"
    GRCh_genes = pd.read_csv(GRCh_downloaded_data, sep='\t')
    GRCh_genes['Gene_stable_ID'] = GRCh_genes['Gene_stable_ID'].str.strip()
    GRCh_genes['Gene_name'] = GRCh_genes['Gene_name'].str.strip()

    # HGNC DATA
    HGNC_downloaded_data = data_path + "Downloaded/HGNC/gene_lookup_dictionary.tsv"
    HGNC_genes = pd.read_csv(HGNC_downloaded_data, sep='\t')
    HGNC_genes['ensembl_gene_id'] = HGNC_genes['ensembl_gene_id'].str.strip()
    HGNC_genes['symbol'] = HGNC_genes['symbol'].str.strip()

    # Combine ENSEMBLE and HGNC data
    stable_IDs = pd.concat([GRCh_genes['Gene_stable_ID'], HGNC_genes['ensembl_gene_id']])
    gene_names = pd.concat([GRCh_genes['Gene_name'], HGNC_genes['symbol']])

    for id in pd.unique(stable_IDs):
        idx = indexTable[indexTable['Stable ID'] == id].index
        if not idx.empty:
            indexTable.at[idx[0], 'Gene Name'] = gene_names[stable_IDs == id].values[0]

    # Add in gene locations/positional information (using legacy "HWG nodes.mat")
    # This part assumes you have a Python equivalent of the .mat file loaded as `cur_tbl`
    curtableId2idx = {id: i for i, id in enumerate(cur_tbl.index)}
    gene_info = pd.DataFrame(columns=['Chromosome', 'Start', 'End'])
    for id in indexTable['Stable ID']:
        if id in curtableId2idx:
            idx = curtableId2idx[id]
            gene_info = gene_info.append({
                'Chromosome': cur_tbl.loc[idx, 'Chromosome'],
                'Start': cur_tbl.loc[idx, 'GeneStart'],
                'End': cur_tbl.loc[idx, 'GeneEnd']
            }, ignore_index=True)
        else:
            gene_info = gene_info.append({'Chromosome': 0, 'Start': 0, 'End': 0}, ignore_index=True)

    indexTable = pd.concat([indexTable, gene_info], axis=1)

    return indexTable
##extra function needed for buildAdjacencyMatrix()
def list_HuRI():
    data_path = load_data_path()
### CODES TO BUILD HARDWIRED GENOME A MATRIX ###
def buildAdjacencyMatrix():
    """This function assembles the STRING and HuRI downloaded data into an adjacency matrix.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/buildHWGobj.m
    """
    pass
