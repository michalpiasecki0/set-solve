
# set-solve

## Description
You play Set.  
You have fun.  
You cannot find set on table.  
No fun.  
Take photo, let machine rescue you.  

## Installation 

1. Clone repo   
`git clone git@github.com:michalpiasecki0/set-solve.git`
2. Change path to repo  
`cd set-solve`
2. Create virtual env && install packages  
(**Linux**)
```
python -m venv .venv  
source .venv/bin/activate
pip install -r requirements.txt
```
**Windows**
```
python -m venv .venv  
.venv/Scripts/activate
pip install -r requirements.txt
```

## TO DO (steps):
* ~~write segmentation module~~
* ~~label dataset with segmented cards using LabelStudio~~
* ~~transform labelled dataset + write dataloader~~ 
* write classifier for each feature
* write tests for segmentation
* write set logic


# Kasia to do
* in your local repo do following
1. switch to main branch  
`git checkout main`   
2. update branch with newest changes  
`git pull`  
3. create new branch in LOCAL repository, by taking branch from remote  
`git checkout -b kasia-practice origin/kasia-practice`
4. now you are LOCALLY on branch `kasia-practice` where you can write functions (you can verify with `git status`)
5. install requirements again as i added some libs
`pip install -r requirements.txt`  
* in file `src/solve.py`, you will find classes which implements SetLogic
* I left two methods `find_sets.py` `is_set.py` for you to implement (#TODO)  
 In order to test solutions there are tests in `tests/test_solver.py`.  
 you can test by running `pytest` from repo root path.

