import os
import urllib.request
import gzip
import shutil
### CODES TO DOWNLOAD DATA ###
def load_data_path():
    filepath = os.path.dirname(os.path.abspath(__file__))
    # Go to base path of the repo and add Data/
    data_path = os.path.join(filepath[:-10], "Data/")
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
    pass

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
    pass


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
    pass


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
    pass


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
    pass


### CODES TO BUILD INDEX TABLE ###
def buildIndexTable():
    """This function should build an index table to map each entry of the A matrix to gene names.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/buildIndexTable.m
    """
    pass

### CODES TO BUILD HARDWIRED GENOME A MATRIX ###
def buildAdjacencyMatrix():
    """This function assembles the STRING and HuRI downloaded data into an adjacency matrix.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/buildHWGobj.m
    """
    pass
