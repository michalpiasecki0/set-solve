# to do
* install repo according to installation guide
* 
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
* install repo according to instruction
* read `main.py`, understand what's happening and try to run with example input:
(argparse give you arguments)  
Hint `python main.py --img-path <your_argument> --output-path <your_argument>`  
You should get segmented set cards in output folder.  
* switch to your branch, make some modification in `classification.py` file and push it to your remote branch on repo   