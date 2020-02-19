# EditNTS
This repo contains the code for our paper "[EditNTS: An Neural Programmer-Interpreter Model for Sentence Simplification through Explicit Editing](https://arxiv.org/abs/1906.08104)" accepted at ACL 2019. Please contact me at yue.dong2@mail.mcgill.ca for any question.

Please cite this paper if you use our code or system output.
```
@inproceedings{dong2019editnts,
  title={EditNTS: An Neural Programmer-Interpreter Model for Sentence Simplification through Explicit Editing},
  author={Dong, Yue and Li, Zichao and Rezagholizadeh, Mehdi and Cheung, Jackie Chi Kit},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages = {3393--3402},
  year={2019}
}
```
## Our Test Output:
https://drive.google.com/file/d/1zT6pNO_zixZLgtwqxaiErzdeiAKrALzt/view?usp=sharing

### Installation
Our code is written with python 2.7. 

Our code requires PyTorch version >= 0.4.1. Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

You will need to first process the data using data_preprocess.py and then run our model through main.py. 

### A few notes on the Data Processing:

Please first call the data_processing.py and make sure the processed dataframe has the following columns:
['comp_tokens', 'simp_tokens','comp_ids','simp_ids', 'comp_pos_tags', 'comp_pos_ids', edit_labels','new_edit_ids']

This can be done by the following steps:

1.process_raw_data(comp_txt, simp_txt)

2.editnet_data_to_editnetID(df,output_path).

after getting the processed data (in a dataframe), you can run the training through main.py. In the paper, we filtered out the rows where the source sentence and the target sentence are identical to encourge editing, you can do this by adding a line at line 41 in data_processing.py: 

comp_txt,simp_txt=unzip([(i[0],i[1]) for i in zip(comp_txt,simp_txt)] if i[0] != i[1]]).



