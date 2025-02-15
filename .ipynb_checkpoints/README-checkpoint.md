# HardwiredGenome-Python
HardwiredGenome Project (v2)

To build the HWG at the desired threshold, run this code:
```
from HWG.build import *
import pandas as pd
import os
import numpy as np

# TODO: set your value of threshold
HWG = loadHWG(thresh=thresh)
```

This will return a python dictionary in the following form:
```
HWG = {
    'thresh': thresh,              # threshold used to construct the network
    'A': A_HWG,                    # adjacency matrix
    'geneIndexTable': indexTable,  # align the network to the gene names
}
```


To access it once you have already built it, you can run the same line of code, and it won't redownload/rebuild it if it exists:
```
HWG = loadHWG(thresh=thresh)
```

You can also save the HWG object in other formats however you like.


Joshua Pickard and Yuchen Shao wrote this code in the summer of 2024.