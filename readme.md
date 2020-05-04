# CombOpt Zero
CombOpt Zero is a general-purpose solver based on AlphaGo Zero for combinatorial problems on graphs.
Paper: [Solving NP-Hard Problems on Graphs with Extended AlphaGo Zero](https://arxiv.org/abs/1905.11623)

You can try *MinimumVertexCover*, *MaximumIndependentSet*, *FeedbackVertexSet*, *MaxCut* and *MaximumClique*, by running the code in this repository.

## Try on Docker
Install [Docker](https://docs.docker.com/get-docker/) and just run `docker/install.sh`, `docker/train.sh` and `docker/eval.sh`!

#### Note
- By default, it solves *MaximumClique*
- Change `docker/config.sh` and `{problem}/config.sh` for other settings
- Hyperparameters are modified so that the training and evaluation can be executed quickly on laptops without GPUs
- But still, **it will obtain pretty good solutions for real-world graphs of thousands of nodes even if trained for only a few minutes** (Try and check it by yourself!)
- `docker/train.sh` may yield some errors, possibly due to the file system of Docker. Please refer to [FAQs](https://github.com/xuzijian629/combopt-zero/wiki/FAQs).

## Build and Run
If you just want to try on docker, please ignore this section.

1. Download LibTorch from https://pytorch.org/  
Download version `1.3.0`. Newer version may cause errors. If you use Linux, download `Pre-cxx11 ABI` version.

2. Build library  
Please also refer to `docker/install.sh` if you have some problem.
```bash
$ cd max-clique/lib
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

3. Generate scripts  
First, modify hyperparameters and other parameters in `{problem}/config.sh`.
Then,
```bash
$ cd max-clique
# create two scripts for training and evalution, named t_sample.sh and e_sample.sh, based on config.sh
$ echo sample | python script_generator.py
```

4. Start training  
You can terminate the training anytime. If you want to restart the training, just run the same command again.
Model files and temporary files are stored in `{problem}/results/{configuration}/`.
```bash
$ cd max-clique
$ ./t_sample.sh
```

5. Start evaluation  
```bash
$ cd max-clique
$ ./e_sample.sh
```


## Dataset
All the test graphs used in our experiments are in `test_graphs/`. Some of them are collected from [Dimacs Vertex Cover instances](https://turing.cs.hbg.psu.edu/txn131/vertex_cover.html) and http://networkrepository.com/.

## Links
- Prototype for *MaximumIndependentSet* in Python: https://github.com/knshnb/MIS_solver

## Cite
Please cite [our paper](https://arxiv.org/abs/1905.11623) if you use our code in your work:

```
@article{Xu/Abe/2020,
    title={Solving NP-Hard Problems on Graphs with Extended AlphaGo Zero},
    author={Zijian Xu and Kenshin Abe and Issei Sato and Masashi Sugiyama},
    journal={arXiv preprint arXiv:1905.11623},
    year={2020}
}
```
