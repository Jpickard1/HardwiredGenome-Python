import os
import urllib.request

### CODES TO DOWNLOAD DATA ###
def download_data():
    """This function should call the methods to download the STRING and HURI datasets.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m
    """
    pass

def download_string():
    """This function should download the STRING database to the computer.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m lines 28-34
    """
    pass

def download_huri():
    """This function should download the HuRI database to the computer.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m lines 18-26
    """
    # Define the path for downloaded data
    downloaded_data_path = "path/to/your/downloaded/data/"
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
    
    print("HuRI Downloaded")


def download_HGNC():
    """This function should download the HGNC database to the computer.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m lines 46-54
    """
    pass

def download_HumanTF():
    """This function should download the Human Transcription Factors database to the computer.
    See: https://github.com/Jpickard1/HardwiredGenome/blob/master/Code/data_handling/build/data_download.m lines 56-64
    """
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
