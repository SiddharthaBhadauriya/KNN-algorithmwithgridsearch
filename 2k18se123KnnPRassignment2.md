We will import all libraries whichever needed, else depending on whenever needed


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import scale, normalize ,StandardScaler
```

As we already saw, that to convert txt to numpy array, we can use genfromtxt, so we will use it here from after downloading our dataset from given source.


```python
nh = np.genfromtxt('sat1.trn')
print (nh.shape)
```

    (4435, 37)
    

Now, for converting it to dataframe, we need to decide the names of their respective columns, so we did


```python
names = ['wr1','wr2','wr3','wr4','wr5','wr6','wr7','wr8','wr9','wr10','wr11','wr12','wr13','wr14','wr15','wr16','wr17','wr18','wr19','wr20','wr21','wr22','wr23','wr24','wr25','wr26','wr27','wr28','wr29','wr30','wr31','wr32','wr33','wr34','wr35','wr36','wr37']
df = pd.DataFrame(nh, columns = names)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wr1</th>
      <th>wr2</th>
      <th>wr3</th>
      <th>wr4</th>
      <th>wr5</th>
      <th>wr6</th>
      <th>wr7</th>
      <th>wr8</th>
      <th>wr9</th>
      <th>wr10</th>
      <th>...</th>
      <th>wr28</th>
      <th>wr29</th>
      <th>wr30</th>
      <th>wr31</th>
      <th>wr32</th>
      <th>wr33</th>
      <th>wr34</th>
      <th>wr35</th>
      <th>wr36</th>
      <th>wr37</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>92.0</td>
      <td>115.0</td>
      <td>120.0</td>
      <td>94.0</td>
      <td>84.0</td>
      <td>102.0</td>
      <td>106.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>102.0</td>
      <td>...</td>
      <td>104.0</td>
      <td>88.0</td>
      <td>121.0</td>
      <td>128.0</td>
      <td>100.0</td>
      <td>84.0</td>
      <td>107.0</td>
      <td>113.0</td>
      <td>87.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>84.0</td>
      <td>102.0</td>
      <td>106.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>102.0</td>
      <td>102.0</td>
      <td>83.0</td>
      <td>80.0</td>
      <td>102.0</td>
      <td>...</td>
      <td>100.0</td>
      <td>84.0</td>
      <td>107.0</td>
      <td>113.0</td>
      <td>87.0</td>
      <td>84.0</td>
      <td>99.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84.0</td>
      <td>102.0</td>
      <td>102.0</td>
      <td>83.0</td>
      <td>80.0</td>
      <td>102.0</td>
      <td>102.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>94.0</td>
      <td>...</td>
      <td>87.0</td>
      <td>84.0</td>
      <td>99.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>99.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80.0</td>
      <td>102.0</td>
      <td>102.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>94.0</td>
      <td>102.0</td>
      <td>79.0</td>
      <td>80.0</td>
      <td>94.0</td>
      <td>...</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>99.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>103.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84.0</td>
      <td>94.0</td>
      <td>102.0</td>
      <td>79.0</td>
      <td>80.0</td>
      <td>94.0</td>
      <td>98.0</td>
      <td>76.0</td>
      <td>80.0</td>
      <td>102.0</td>
      <td>...</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>103.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>107.0</td>
      <td>109.0</td>
      <td>87.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4430</th>
      <td>56.0</td>
      <td>64.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>71.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>68.0</td>
      <td>75.0</td>
      <td>...</td>
      <td>92.0</td>
      <td>66.0</td>
      <td>83.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>66.0</td>
      <td>87.0</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4431</th>
      <td>64.0</td>
      <td>71.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>68.0</td>
      <td>75.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>71.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>96.0</td>
      <td>66.0</td>
      <td>87.0</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>63.0</td>
      <td>87.0</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4432</th>
      <td>68.0</td>
      <td>75.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>71.0</td>
      <td>87.0</td>
      <td>108.0</td>
      <td>88.0</td>
      <td>71.0</td>
      <td>91.0</td>
      <td>...</td>
      <td>89.0</td>
      <td>63.0</td>
      <td>87.0</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>70.0</td>
      <td>100.0</td>
      <td>104.0</td>
      <td>85.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4433</th>
      <td>71.0</td>
      <td>87.0</td>
      <td>108.0</td>
      <td>88.0</td>
      <td>71.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>81.0</td>
      <td>76.0</td>
      <td>95.0</td>
      <td>...</td>
      <td>89.0</td>
      <td>70.0</td>
      <td>100.0</td>
      <td>104.0</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>91.0</td>
      <td>104.0</td>
      <td>85.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4434</th>
      <td>71.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>81.0</td>
      <td>76.0</td>
      <td>95.0</td>
      <td>108.0</td>
      <td>88.0</td>
      <td>80.0</td>
      <td>95.0</td>
      <td>...</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>91.0</td>
      <td>104.0</td>
      <td>85.0</td>
      <td>63.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>81.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>4435 rows × 37 columns</p>
</div>




```python
y = df['wr37'].values
y[-6:-5]
```




    array([5.])




```python
dff = df.drop(columns=['wr37'])
dff
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wr1</th>
      <th>wr2</th>
      <th>wr3</th>
      <th>wr4</th>
      <th>wr5</th>
      <th>wr6</th>
      <th>wr7</th>
      <th>wr8</th>
      <th>wr9</th>
      <th>wr10</th>
      <th>...</th>
      <th>wr27</th>
      <th>wr28</th>
      <th>wr29</th>
      <th>wr30</th>
      <th>wr31</th>
      <th>wr32</th>
      <th>wr33</th>
      <th>wr34</th>
      <th>wr35</th>
      <th>wr36</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>92.0</td>
      <td>115.0</td>
      <td>120.0</td>
      <td>94.0</td>
      <td>84.0</td>
      <td>102.0</td>
      <td>106.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>102.0</td>
      <td>...</td>
      <td>134.0</td>
      <td>104.0</td>
      <td>88.0</td>
      <td>121.0</td>
      <td>128.0</td>
      <td>100.0</td>
      <td>84.0</td>
      <td>107.0</td>
      <td>113.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>84.0</td>
      <td>102.0</td>
      <td>106.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>102.0</td>
      <td>102.0</td>
      <td>83.0</td>
      <td>80.0</td>
      <td>102.0</td>
      <td>...</td>
      <td>128.0</td>
      <td>100.0</td>
      <td>84.0</td>
      <td>107.0</td>
      <td>113.0</td>
      <td>87.0</td>
      <td>84.0</td>
      <td>99.0</td>
      <td>104.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84.0</td>
      <td>102.0</td>
      <td>102.0</td>
      <td>83.0</td>
      <td>80.0</td>
      <td>102.0</td>
      <td>102.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>94.0</td>
      <td>...</td>
      <td>113.0</td>
      <td>87.0</td>
      <td>84.0</td>
      <td>99.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>99.0</td>
      <td>104.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80.0</td>
      <td>102.0</td>
      <td>102.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>94.0</td>
      <td>102.0</td>
      <td>79.0</td>
      <td>80.0</td>
      <td>94.0</td>
      <td>...</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>99.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>103.0</td>
      <td>104.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84.0</td>
      <td>94.0</td>
      <td>102.0</td>
      <td>79.0</td>
      <td>80.0</td>
      <td>94.0</td>
      <td>98.0</td>
      <td>76.0</td>
      <td>80.0</td>
      <td>102.0</td>
      <td>...</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>84.0</td>
      <td>103.0</td>
      <td>104.0</td>
      <td>79.0</td>
      <td>79.0</td>
      <td>107.0</td>
      <td>109.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4430</th>
      <td>56.0</td>
      <td>64.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>64.0</td>
      <td>71.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>68.0</td>
      <td>75.0</td>
      <td>...</td>
      <td>108.0</td>
      <td>92.0</td>
      <td>66.0</td>
      <td>83.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>66.0</td>
      <td>87.0</td>
      <td>104.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>4431</th>
      <td>64.0</td>
      <td>71.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>68.0</td>
      <td>75.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>71.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>66.0</td>
      <td>87.0</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>63.0</td>
      <td>87.0</td>
      <td>104.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>4432</th>
      <td>68.0</td>
      <td>75.0</td>
      <td>108.0</td>
      <td>96.0</td>
      <td>71.0</td>
      <td>87.0</td>
      <td>108.0</td>
      <td>88.0</td>
      <td>71.0</td>
      <td>91.0</td>
      <td>...</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>63.0</td>
      <td>87.0</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>70.0</td>
      <td>100.0</td>
      <td>104.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>4433</th>
      <td>71.0</td>
      <td>87.0</td>
      <td>108.0</td>
      <td>88.0</td>
      <td>71.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>81.0</td>
      <td>76.0</td>
      <td>95.0</td>
      <td>...</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>70.0</td>
      <td>100.0</td>
      <td>104.0</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>91.0</td>
      <td>104.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>4434</th>
      <td>71.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>81.0</td>
      <td>76.0</td>
      <td>95.0</td>
      <td>108.0</td>
      <td>88.0</td>
      <td>80.0</td>
      <td>95.0</td>
      <td>...</td>
      <td>104.0</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>91.0</td>
      <td>104.0</td>
      <td>85.0</td>
      <td>63.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>81.0</td>
    </tr>
  </tbody>
</table>
<p>4435 rows × 36 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dff,y,test_size=0.3,random_state=1,stratify = y)
```


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=12, p=2,
                         weights='uniform')




```python
knn.predict(X_test)[:-5]
```




    array([2., 2., 2., ..., 3., 7., 3.])




```python
knn.score(X_test,y_test)
```




    0.8888054094665665




```python
from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=4)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
```

    [0.88566828 0.90660225 0.88405797 0.88727858 0.9016129 ]
    cv_scores mean:0.8930439977144043
    


```python
!pip install scikit-learn==0.19.2
```

    Collecting scikit-learn==0.19.2
      Downloading scikit_learn-0.19.2-cp37-cp37m-win_amd64.whl (4.4 MB)
    Installing collected packages: scikit-learn
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 0.22.1
        Uninstalling scikit-learn-0.22.1:
          Successfully uninstalled scikit-learn-0.22.1
    

    ERROR: Could not install packages due to an EnvironmentError: [WinError 5] Access is denied: 'c:\\users\\siddhartha\\anaconda3\\lib\\site-packages\\~klearn\\cluster\\_dbscan_inner.cp37-win_amd64.pyd'
    Consider using the `--user` option or check the permissions.
    
    


```python

from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
hj = list(range(1,25))
param_grid = {'n_neighbors': hj}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5, scoring = 'accuracy')
#print (knn_gscv.error)
#fit model to data
knn_gscv.fit(dff, y)
#pred = knn_gscv.predict(X_test)
```


```python
knn_gscv.score(dff,y)
```




    0.9129650507328072




```python
#knn_gscv.grid_scores_
scores = knn_gscv.cv_results_#['mean_test_score'].reshape(-1, 3).T
scores
```




    {'mean_fit_time': array([0.04969759, 0.04938774, 0.0491118 , 0.05166674, 0.04808106,
            0.04743338, 0.04817986, 0.04739723, 0.05210581, 0.04985657,
            0.05197654, 0.04892497, 0.04698992, 0.05220428, 0.05388923,
            0.05013547, 0.04905376, 0.05076447, 0.05322652, 0.05011048,
            0.04919972, 0.04853554, 0.04962115, 0.0492135 ]),
     'std_fit_time': array([0.00329176, 0.00121255, 0.00091021, 0.00431232, 0.0054341 ,
            0.00094341, 0.00142557, 0.00995209, 0.00404321, 0.00469363,
            0.00537005, 0.00122974, 0.00578359, 0.00659511, 0.00726927,
            0.0022556 , 0.0040493 , 0.00330767, 0.00638487, 0.00203251,
            0.00062535, 0.00159689, 0.00437775, 0.00121475]),
     'mean_score_time': array([0.19537063, 0.20252428, 0.21125169, 0.21262507, 0.22162161,
            0.22767005, 0.22354426, 0.22480817, 0.23137698, 0.230057  ,
            0.22964826, 0.23827243, 0.23360348, 0.23410048, 0.23706636,
            0.239995  , 0.24137807, 0.24421463, 0.24148846, 0.25469222,
            0.24752502, 0.25154619, 0.25380626, 0.2552237 ]),
     'std_score_time': array([0.00445718, 0.00643953, 0.01179891, 0.00937483, 0.01212292,
            0.01045849, 0.01016228, 0.01161049, 0.01484658, 0.01906214,
            0.00625773, 0.00744903, 0.00848363, 0.00976614, 0.00898889,
            0.01198547, 0.00733546, 0.01756723, 0.01012442, 0.01692792,
            0.01327194, 0.01938497, 0.01021886, 0.01268927]),
     'param_n_neighbors': masked_array(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'n_neighbors': 1},
      {'n_neighbors': 2},
      {'n_neighbors': 3},
      {'n_neighbors': 4},
      {'n_neighbors': 5},
      {'n_neighbors': 6},
      {'n_neighbors': 7},
      {'n_neighbors': 8},
      {'n_neighbors': 9},
      {'n_neighbors': 10},
      {'n_neighbors': 11},
      {'n_neighbors': 12},
      {'n_neighbors': 13},
      {'n_neighbors': 14},
      {'n_neighbors': 15},
      {'n_neighbors': 16},
      {'n_neighbors': 17},
      {'n_neighbors': 18},
      {'n_neighbors': 19},
      {'n_neighbors': 20},
      {'n_neighbors': 21},
      {'n_neighbors': 22},
      {'n_neighbors': 23},
      {'n_neighbors': 24}],
     'split0_test_score': array([0.8410372 , 0.85231116, 0.85005637, 0.84780158, 0.85794814,
            0.87034949, 0.87147689, 0.87598647, 0.8692221 , 0.87824126,
            0.87824126, 0.87485908, 0.87711387, 0.87485908, 0.87373168,
            0.87147689, 0.87147689, 0.87034949, 0.8692221 , 0.8692221 ,
            0.87034949, 0.86583991, 0.8692221 , 0.87034949]),
     'split1_test_score': array([0.81848929, 0.83201804, 0.83089064, 0.8421646 , 0.83765502,
            0.8421646 , 0.83540023, 0.83878241, 0.83878241, 0.83990981,
            0.8410372 , 0.84441939, 0.84780158, 0.84441939, 0.843292  ,
            0.8410372 , 0.84441939, 0.84441939, 0.84667418, 0.84667418,
            0.84892897, 0.84554679, 0.84554679, 0.85005637]),
     'split2_test_score': array([0.85907554, 0.88613303, 0.87034949, 0.88049605, 0.87485908,
            0.88275085, 0.87711387, 0.88275085, 0.88275085, 0.88275085,
            0.88275085, 0.88613303, 0.88387824, 0.88726043, 0.88387824,
            0.88162345, 0.87936866, 0.88275085, 0.88049605, 0.87824126,
            0.87598647, 0.87598647, 0.87711387, 0.87936866]),
     'split3_test_score': array([0.82074408, 0.80834273, 0.82187148, 0.82074408, 0.82976325,
            0.82638106, 0.82638106, 0.82863585, 0.82863585, 0.83427283,
            0.83201804, 0.83201804, 0.82863585, 0.82750846, 0.82750846,
            0.82638106, 0.82750846, 0.82074408, 0.82863585, 0.82638106,
            0.82750846, 0.82638106, 0.82525366, 0.82638106]),
     'split4_test_score': array([0.78241263, 0.78466742, 0.79932356, 0.80045096, 0.80383315,
            0.80721533, 0.81172492, 0.81848929, 0.81961669, 0.82074408,
            0.82525366, 0.83089064, 0.82525366, 0.82074408, 0.82638106,
            0.82863585, 0.82976325, 0.82638106, 0.82976325, 0.82412627,
            0.82187148, 0.82299887, 0.82187148, 0.81961669]),
     'mean_test_score': array([0.82435175, 0.83269448, 0.83449831, 0.83833145, 0.84081172,
            0.84577227, 0.84441939, 0.84892897, 0.84780158, 0.85118377,
            0.8518602 , 0.85366404, 0.85253664, 0.85095829, 0.85095829,
            0.84983089, 0.85050733, 0.84892897, 0.85095829, 0.84892897,
            0.84892897, 0.84735062, 0.84780158, 0.84915445]),
     'std_test_score': array([0.02565315, 0.03504442, 0.02423665, 0.02693131, 0.0242995 ,
            0.02773852, 0.02559958, 0.02575797, 0.02417994, 0.02477194,
            0.02395391, 0.02269848, 0.02418415, 0.02605626, 0.02372786,
            0.02260871, 0.02130271, 0.02419046, 0.02084189, 0.02190748,
            0.0218261 , 0.02096592, 0.02237817, 0.02346934]),
     'rank_test_score': array([24, 23, 22, 21, 20, 18, 19, 11, 15,  4,  3,  1,  2,  5,  5,  9,  8,
            14,  5, 11, 11, 17, 15, 10])}




```python
knn_gscv.best_params_
```




    {'n_neighbors': 12}




```python
error_rate = []
for i in range(2,25):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
```


```python
plt.figure(figsize=(10,6))
plt.plot(range(2,25),error_rate,color='blue', linestyle='dashed', marker='o',
markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```




    Text(0, 0.5, 'Error Rate')




![png](output_19_1.png)



```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=12, p=2,
                         weights='uniform')




```python
knn.score(X_test,y_test)
```




    0.8888054094665665




```python
knn.score(X_train,y_train)
```




    0.8927190721649485




```python

```
