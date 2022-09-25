# Assignment 1

## Installation
The code in this assignment is tested on Python 3.8
Easy installation of the virtual environment:
```
git clone https://github.com/aali343/as1
cd as1
bash install.sh
```
Required packages:
scikit-learn 
matplotlib
pandas


## How to run all scripts

```
source py38/bin/activate

python knn.py
python decision_tree_pruned.py
python boosting_pruned.py
python neural_net.py
python svm.py
```

## Output
Learning curves and logs are present in the directory corresponding to the algorithm name
Example:
```
python knn.py
```
Will produce the output in `./knn/`