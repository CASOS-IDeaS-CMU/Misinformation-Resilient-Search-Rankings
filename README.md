# Misinformation Resilient Search Rankings
### Introduction
These scripts implement and evaluate search ranking interventions based on the research paper "Misinformation Resilient Search Rankings with Webgraph-based Interventions". We use the [NewsSEO](https://doi.org/10.1184/R1/25174193.v1) and [CommonCrawl](https://data.commoncrawl.org/projects/hyperlinkgraph/cc-main-2023-may-sep-nov/index.html) webgraph datasets. If you use, extend or build upon this project, please cite the following paper:
```
@article{carragher2024misinformation,
  title={Misinformation Resilient Search Rankings with Webgraph-based Interventions},
  author={Carragher, Peter and Williams, Evan M and Carley, Kathleen M},
  journal={ACM Transactions on Intelligent Systems and Technology},
  year={2024}
}
```

### Inputs
* Follow [the readme](data_collection/README.md) to populate the data directory with the [NewsSEO dataset](https://kilthub.cmu.edu/articles/dataset/Dataset_for_Detection_and_Discovery_of_Misinformation_Sources_using_Attributed_Webgraphs_/25174193/1)
* Webgraph data & SEO attributes have been pulled from ahrefs.com and commoncrawl.org
* Labels have been scraped from mediabiasfactcheck.com using this [open-source scraper](https://github.com/CASOS-IDeaS-CMU/media_bias_fact_check_data_collection)
* For more information, see the original paper ["Detection and Discovery of Misinformation Sources"](https://arxiv.org/abs/2401.02379)

### Environment Setup
```
conda create --name mrsr --file requirements.txt
conda activate mrsr

mkdir <ROOT_DIR>/data && cd <ROOT_DIR>/data
wget https://kilthub.cmu.edu/ndownloader/articles/25174193/versions/1 -O dataset.zip
7z x dataset.zip

mkdir <ROOT_DIR>/results/multiplicity
cd <ROOT_DIR>/interventions && jupyter nbconvert --to notebook --inplace --execute multiplicity.ipynb
python3 bias_removal.py
cd ../regressions && python3 intervention_eval.py
```

### Outputs
* analysis: produces figures for the analysis of backlink distributions, political leaning, and news network traffic
* data_collection: scripts for collecting data from the Ahrefs API
* gnns: scripts for training GNN models to predict traffic and ranking based on webgraphs
* interventions: source code for intervention design of backlink removal and multiplicity interventions
* regressions: source code for fitting regression models to predict traffic and evaluate interventions 

### Acknowledgements
Thanks to the authors of CommonCrawl's [Webgraph utilities](https://github.com/commoncrawl/cc-webgraph), which we base our large-scale experiments on. Thanks also to the authors of the [WebGraph framework](https://webgraph.di.unimi.it/), an [open source](//github.com/vigna/webgraph) framework which is used to process the CommonCrawl graphs and compute pagerank.

### License
BSD 3-Clause License

Copyright (c) 2024, Peter Carragher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

