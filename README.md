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
@InProceedings{Meng_2021_BMVC,
    author    = {Meng, Yanda and Zhang, Hongrun and Gao, Dongxu and Zhao, Yitian and Yang, Xiaoyun and Qian, Xuesheng and Huang, Xiaowei and Zheng, Yalin},
    title     = {BI-GConv: Boundary-Aware Input-Dependent Graph Convolution for Biomedical Image Segmentation},
    booktitle = {BMVC},
    year      = {2021},
}

```


