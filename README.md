# BMVC 2021
# BI-GConv: Boundary-Aware Input-Dependent Graph Convolution for Biomedical Image Segmentation 

---------------------------
Necassary Dependencies: PyTorch 1.2.0  Python 3.6
        


# -test-

--Download best_model.pth and put it into ../model/oc_od/ODOC_BMVC_48_bs_beta_0.1_base_lr_0.006/

Link: https://drive.google.com/file/d/1GhBDphV4VUQ7KdxwC6kgzQO3Q-uSq3dn/view?usp=sharing

--The index of test data is in oc_od/h5py_all/test.txt

--Prepare the Test data , then put them into your_folder/oc_od/h5py_all

--Run the test_odoc.py


# -train-

--The index of train_val data is in oc_od/h5py_all/train.txt

--Prepare the Train data, then put them into your_folder/oc_od/h5py_all

--Run the train_odoc.py





# Citation
If you find our work useful or our work gives you any insights, please cite:
```
@inproceedings{DBLP:conf/bmvc/MengZGZYQ0Z21,
  author    = {Yanda Meng and
               Hongrun Zhang and
               Dongxu Gao and
               Yitian Zhao and
               Xiaoyun Yang and
               Xuesheng Qian and
               Xiaowei Huang and
               Yalin Zheng},
  title     = {{BI-GCN:} Boundary-Aware Input-Dependent Graph Convolution Network
               for Biomedical Image Segmentation},
  booktitle = {32nd British Machine Vision Conference 2021, {BMVC} 2021, Online,
               November 22-25, 2021},
  pages     = {223},
  publisher = {{BMVA} Press},
  year      = {2021},
  url       = {https://www.bmvc2021-virtualconference.com/assets/papers/0097.pdf},
  timestamp = {Wed, 22 Jun 2022 16:52:45 +0200},
  biburl    = {https://dblp.org/rec/conf/bmvc/MengZGZYQ0Z21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
