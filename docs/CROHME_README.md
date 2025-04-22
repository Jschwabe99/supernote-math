---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: string
  splits:
  - name: train
    num_bytes: 584818053.75
    num_examples: 8834
  - name: '2014'
    num_bytes: 67592822.0
    num_examples: 986
  - name: '2016'
    num_bytes: 81409074.625
    num_examples: 1147
  - name: '2019'
    num_bytes: 91018292.125
    num_examples: 1199
  download_size: 813177600
  dataset_size: 824838242.5
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: '2014'
    path: data/2014-*
  - split: '2016'
    path: data/2016-*
  - split: '2019'
    path: data/2019-*
---
