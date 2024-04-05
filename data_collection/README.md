### Setup
Download and unzip dataset from [kilthub](https://kilthub.cmu.edu/articles/dataset/Dataset_for_Detection_and_Discovery_of_Misinformation_Sources_using_Attributed_Webgraphs_/25174193/1):
```
mkdir <ROOT_DIR>/data && cd <ROOT_DIR>/data
wget https://kilthub.cmu.edu/ndownloader/articles/25174193/versions/1 -O dataset.zip
7z x dataset.zip 
```

Once downloaded you can view the dataset readme for more information about the networks and attributes involved.

### Sources
Data were pulled from the Ahrefs API using the [RAhrefs package](https://github.com/Leszek-Sieminski/RAhrefs), with labels pulled using the [MBFC Scraper](https://github.com/CASOS-IDeaS-CMU/media_bias_fact_check_data_collection)

### Reproducability
Webgraphs are dynamic, and so attempts to reproduce this dataset will have more up-to-date attributes, backlinks, and outlinks, reflecting changes to the structure of the news domains since the time of this study.
Relevant scripts allow this research to be reproduced with values dependent upon these changes:
* [ahref_backlinks.R](ahref_backlinks.R): fetch backlinks for a given list of domains
* [ahref_outlinks.R](ahref_outlinks.R): fetch outlinks for a given list of domains
* [ahref_nodes.R](ahref_nodes.R): fetch attributes for a given list of domains