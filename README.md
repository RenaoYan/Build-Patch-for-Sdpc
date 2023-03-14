# Build-Patch-for-Sdpc
Build patches from whole slide images (WSIs) in .sdpc format.

Two approaches (build with/ without .sdpl) to build patches are given for two different platforms (windows/ linux). 

# Environment Setup
- **Windows Platform:** 

`pip install sdpc-win` or download from [sdpc for windows (version 3.0)](https://pypi.org/project/sdpc-win/#files)

- **Linux Platform:**

`pip install sdpc-linux` or download from [sdpc for linux (version 1.5)](https://pypi.org/project/sdpc-linux/#files)

# Problems List
If you found `Patch Saving Finished` occured before `Patch Reading Finished`, you could increase the block time from 10 up to 100 or even more in `img_dict = q.get(block=event.is_set(), timeout=10)`.

# Citing This Work
```
@misc{Yan2023,
  author = {Renao, Yan and Yiqing, Liu and Qiehe, Sun and Jiawen, Li and Lianghui, Zhu and Qiming, He},
  title = {Build-Patch-for-Sdpc},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/RenaoYan/Build-Patch-for-Sdpc}},
  DOI = {10.5281/zenodo.7680346}
}
```
