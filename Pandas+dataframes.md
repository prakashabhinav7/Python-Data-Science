
PANDAS
------


```python
import numpy as np
```


```python
import pandas as pd
```


```python
labels_list = ['a','b','c'] #list
my_data_list = [10,20,30] #list
arr = np.array(my_data_list) #array
dict = {'a':10,'b':20,'c':30}#dictionary
```


```python
pd.Series(data=my_data_list,index=labels_list)#series are indexed and can be called with those indices
```




    a    10
    b    20
    c    30
    dtype: int64




```python
pd.Series(my_data_list,labels_list)#alternate way of creating the series
```




    a    10
    b    20
    c    30
    dtype: int64




```python
pd.Series(dict)#uses the keys as the index
```




    a    10
    b    20
    c    30
    dtype: int64



DATAFRAMES
------


```python
from numpy.random import randn
```


```python
np.random.seed(101)
```


```python
df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
```


```python
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['W'] #This returns the column W, which itself is a series with index 'A''B''C''D''E'. This can be checked by using 
#type(df['W']) which returns the type
```




    A    2.706850
    B    0.651118
    C   -2.018168
    D    0.188695
    E    0.190794
    Name: W, dtype: float64




```python
df[['W','Z']] #Passing a list of columns returns a dataframe(not just a single series obviously)
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
      <th>W</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['new'] = df['W']+df['Z'] #A new column can be added to the dataframe on the fly
```


```python
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>3.210676</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>1.257083</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>-2.607169</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>1.143752</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>0.874303</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('new',axis=1) #The drop statement won't remove the actual table as can be seen below. For that use the 
#inplace=True option with the drop command
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>3.210676</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>1.257083</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>-2.607169</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>1.143752</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>0.874303</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('new',axis=1,inplace=True) #Now the 'new' column has been removed
```


```python
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('E')#drops the E row and we dont need to specify the axis in this case since that is default(0)
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['C'] #The rows are series as well. loc function return label based row
```




    W   -2.018168
    X    0.740122
    Y    0.528813
    Z   -0.589001
    Name: C, dtype: float64




```python
df.iloc[2] #iloc function returns values based on index of the row
```




    W   -2.018168
    X    0.740122
    Y    0.528813
    Z   -0.589001
    Name: C, dtype: float64




```python
df.loc[['A','B'],['X','Z']] #returning a subset of the dataframe using row and column labels
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
      <th>X</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.628133</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.319318</td>
      <td>0.605965</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[[1],[1,3]] #returning a subset of the dataframe using row and column indexes
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
      <th>X</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>B</th>
      <td>-0.319318</td>
      <td>0.605965</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
```


```python
df[df > 0] #conditional selection within dataframes
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>NaN</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['W']>0] #Returns a df filtered based on the value of W being greater than 0
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>-0.848077</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['W']>0]['X'] #We can filter even furhter on the resulting df by stacking commands as done here
```




    A    0.628133
    B   -0.319318
    D   -0.758872
    E    1.978757
    Name: X, dtype: float64




```python
df[(df['W']>0) & (df['Y']>2)] # stacking conditional statements. DONOT use 'and' instead of '&'
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>E</th>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>2.605967</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>



MultiIndex and Index Heirarchy
----------------------------


```python
#Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
heir_index = list(zip(outside,inside))
heir_index = pd.MultiIndex.from_tuples(heir_index)
```


```python
heir_index
```




    MultiIndex(levels=[['G1', 'G2'], [1, 2, 3]],
               labels=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])




```python
df2 = pd.DataFrame(randn(6,2),heir_index,['A','B']) #creating a multi-indexed(heirarchy)dataframe
```


```python
df2
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>0.681209</td>
      <td>1.035125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.031160</td>
      <td>1.939932</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.005187</td>
      <td>-0.741790</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>0.187125</td>
      <td>-0.732845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.382920</td>
      <td>1.482495</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.961458</td>
      <td>-2.141212</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.loc['G1'].loc[2] #grabbing data from the heirarchy 
```




    A   -0.031160
    B    1.939932
    Name: 2, dtype: float64




```python
df2.index.names = ['Groups','Id'] #Renaming the labels for the indexes
df2
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Groups</th>
      <th>Id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">G1</th>
      <th>1</th>
      <td>0.681209</td>
      <td>1.035125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.031160</td>
      <td>1.939932</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.005187</td>
      <td>-0.741790</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">G2</th>
      <th>1</th>
      <td>0.187125</td>
      <td>-0.732845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.382920</td>
      <td>1.482495</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.961458</td>
      <td>-2.141212</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.index.names
```




    FrozenList(['Groups', 'Id'])




```python
df2.xs(1,level='Id') #cross section function grabs the data from G1 and G2 for level 1. This would be a little tricky using loc
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
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>Groups</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>G1</th>
      <td>0.681209</td>
      <td>1.035125</td>
    </tr>
    <tr>
      <th>G2</th>
      <td>0.187125</td>
      <td>-0.732845</td>
    </tr>
  </tbody>
</table>
</div>



Creating DataFrames from dictionaries and Missing Data
-----


```python
dict1 = {'A':[1,2,np.nan],'B':[4,np.nan, np.nan],'C':[1,2,3]}
```


```python
df3 = pd.DataFrame(dict1)
```


```python
df3
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.isnull() #Checks for null
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.dropna()#dropping rows with null values(uses default axis)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.dropna(axis=1)#dropping cols with null values
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
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.dropna(axis=1, thresh=2)#using threshold values for dropping columns based on number of null values
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
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.fillna(value='dummy') #filling values for NULLS in the df. Could use mean,average,etc. to fill these values
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>dummy</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dummy</td>
      <td>dummy</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



GroupBy
---


```python
data_dict = {'Company':['Google','Google','Google','Microsoft','Microsoft','Facebook'],
             'Person':['yum','yo','yun','yin','yam','yum'],
             'Sales':[120,233,342,143,149,544]}
```


```python
df4 = pd.DataFrame(data_dict)
df4
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
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Google</td>
      <td>yum</td>
      <td>120</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Google</td>
      <td>yo</td>
      <td>233</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Google</td>
      <td>yun</td>
      <td>342</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Microsoft</td>
      <td>yin</td>
      <td>143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Microsoft</td>
      <td>yam</td>
      <td>149</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Facebook</td>
      <td>yum</td>
      <td>544</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4.groupby('Company').mean() #Can use other aggregate functions on the grouped by data and also use count,loc,etc.
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
      <th>Sales</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Facebook</th>
      <td>544.000000</td>
    </tr>
    <tr>
      <th>Google</th>
      <td>231.666667</td>
    </tr>
    <tr>
      <th>Microsoft</th>
      <td>146.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4.groupby('Company').describe() #describe() is useful for listing out a lot of details about the data and it can be
#transposed as well to fit the data accordingly
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">Sales</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Facebook</th>
      <td>1.0</td>
      <td>544.000000</td>
      <td>NaN</td>
      <td>544.0</td>
      <td>544.0</td>
      <td>544.0</td>
      <td>544.0</td>
      <td>544.0</td>
    </tr>
    <tr>
      <th>Google</th>
      <td>3.0</td>
      <td>231.666667</td>
      <td>111.006006</td>
      <td>120.0</td>
      <td>176.5</td>
      <td>233.0</td>
      <td>287.5</td>
      <td>342.0</td>
    </tr>
    <tr>
      <th>Microsoft</th>
      <td>2.0</td>
      <td>146.000000</td>
      <td>4.242641</td>
      <td>143.0</td>
      <td>144.5</td>
      <td>146.0</td>
      <td>147.5</td>
      <td>149.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4
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
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Google</td>
      <td>yum</td>
      <td>120</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Google</td>
      <td>yo</td>
      <td>233</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Google</td>
      <td>yun</td>
      <td>342</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Microsoft</td>
      <td>yin</td>
      <td>143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Microsoft</td>
      <td>yam</td>
      <td>149</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Facebook</td>
      <td>yum</td>
      <td>544</td>
    </tr>
  </tbody>
</table>
</div>



Operations
---


```python
len(df4)
```




    6




```python
df4['Company'].unique()
```




    array(['Google', 'Microsoft', 'Facebook'], dtype=object)




```python
df4['Company'].nunique() #number of unique values
```




    3




```python
df4['Company'].value_counts() #number of times each unique values is repeated
```




    Google       3
    Microsoft    2
    Facebook     1
    Name: Company, dtype: int64



Applying your own functions
--


```python
def squared(param):        #UDF to be applied
    return param*param
```


```python
df4['Sales'].apply(squared) #use the apply method to apply the UDF
```




    0     14400
    1     54289
    2    116964
    3     20449
    4     22201
    5    295936
    Name: Sales, dtype: int64




```python
df4['Sales'].apply(lambda x:x*x) #can use lambda functions as well
```




    0     14400
    1     54289
    2    116964
    3     20449
    4     22201
    5    295936
    Name: Sales, dtype: int64




```python
df4.sort_values('Sales') #Still retains the indices
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
      <th>Company</th>
      <th>Person</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Google</td>
      <td>yum</td>
      <td>120</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Microsoft</td>
      <td>yin</td>
      <td>143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Microsoft</td>
      <td>yam</td>
      <td>149</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Google</td>
      <td>yo</td>
      <td>233</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Google</td>
      <td>yun</td>
      <td>342</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Facebook</td>
      <td>yum</td>
      <td>544</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4.pivot_table(values='Sales',index='Person',columns='Company') #using Pivot Tables-could have made a better one :P
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
      <th>Company</th>
      <th>Facebook</th>
      <th>Google</th>
      <th>Microsoft</th>
    </tr>
    <tr>
      <th>Person</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>yam</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>149.0</td>
    </tr>
    <tr>
      <th>yin</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>143.0</td>
    </tr>
    <tr>
      <th>yo</th>
      <td>NaN</td>
      <td>233.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>yum</th>
      <td>544.0</td>
      <td>120.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>yun</th>
      <td>NaN</td>
      <td>342.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Data I/O
--
Requires installs:
conda install sqlalchemy
conda install lxml
conda install html5lib
conda install BeautifulSoup4
conda install xlrd


```python
## From and to csv and excel
```


```python
df5 = pd.read_csv('example') #reading CSVs to DF
df5
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df5.to_csv('output.csv',index=False) #writing CSVs to DF and index needs to be false else it will create an extra index
#column 
```


```python
pd.read_csv('output.csv')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1') #panda treats different sheets in a workbook as single dataframe
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Reading from HTML
```


```python
html_data = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html') #reads all the table elements of 
# the webpage as a list. We take the first element here which is a dataframe
```


```python
df6=html_data[0]
df6
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
      <th>Bank Name</th>
      <th>City</th>
      <th>ST</th>
      <th>CERT</th>
      <th>Acquiring Institution</th>
      <th>Closing Date</th>
      <th>Updated Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Washington Federal Bank for Savings</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>30570</td>
      <td>Royal Savings Bank</td>
      <td>December 15, 2017</td>
      <td>February 21, 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Farmers and Merchants State Bank of Argonia</td>
      <td>Argonia</td>
      <td>KS</td>
      <td>17719</td>
      <td>Conway Bank</td>
      <td>October 13, 2017</td>
      <td>February 21, 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fayette County Bank</td>
      <td>Saint Elmo</td>
      <td>IL</td>
      <td>1802</td>
      <td>United Fidelity Bank, fsb</td>
      <td>May 26, 2017</td>
      <td>July 26, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Guaranty Bank, (d/b/a BestBank in Georgia &amp; Mi...</td>
      <td>Milwaukee</td>
      <td>WI</td>
      <td>30003</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>May 5, 2017</td>
      <td>March 22, 2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>First NBC Bank</td>
      <td>New Orleans</td>
      <td>LA</td>
      <td>58302</td>
      <td>Whitney Bank</td>
      <td>April 28, 2017</td>
      <td>December 5, 2017</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Proficio Bank</td>
      <td>Cottonwood Heights</td>
      <td>UT</td>
      <td>35495</td>
      <td>Cache Valley Bank</td>
      <td>March 3, 2017</td>
      <td>March 7, 2018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Seaway Bank and Trust Company</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>19328</td>
      <td>State Bank of Texas</td>
      <td>January 27, 2017</td>
      <td>May 18, 2017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Harvest Community Bank</td>
      <td>Pennsville</td>
      <td>NJ</td>
      <td>34951</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>January 13, 2017</td>
      <td>May 18, 2017</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Allied Bank</td>
      <td>Mulberry</td>
      <td>AR</td>
      <td>91</td>
      <td>Today's Bank</td>
      <td>September 23, 2016</td>
      <td>September 25, 2017</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Woodbury Banking Company</td>
      <td>Woodbury</td>
      <td>GA</td>
      <td>11297</td>
      <td>United Bank</td>
      <td>August 19, 2016</td>
      <td>June 1, 2017</td>
    </tr>
    <tr>
      <th>10</th>
      <td>First CornerStone Bank</td>
      <td>King of Prussia</td>
      <td>PA</td>
      <td>35312</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>May 6, 2016</td>
      <td>September 6, 2016</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Trust Company Bank</td>
      <td>Memphis</td>
      <td>TN</td>
      <td>9956</td>
      <td>The Bank of Fayette County</td>
      <td>April 29, 2016</td>
      <td>September 6, 2016</td>
    </tr>
    <tr>
      <th>12</th>
      <td>North Milwaukee State Bank</td>
      <td>Milwaukee</td>
      <td>WI</td>
      <td>20364</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>March 11, 2016</td>
      <td>March 13, 2017</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Hometown National Bank</td>
      <td>Longview</td>
      <td>WA</td>
      <td>35156</td>
      <td>Twin City Bank</td>
      <td>October 2, 2015</td>
      <td>February 19, 2018</td>
    </tr>
    <tr>
      <th>14</th>
      <td>The Bank of Georgia</td>
      <td>Peachtree City</td>
      <td>GA</td>
      <td>35259</td>
      <td>Fidelity Bank</td>
      <td>October 2, 2015</td>
      <td>July 9, 2018</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Premier Bank</td>
      <td>Denver</td>
      <td>CO</td>
      <td>34112</td>
      <td>United Fidelity Bank, fsb</td>
      <td>July 10, 2015</td>
      <td>February 20, 2018</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Edgebrook Bank</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>57772</td>
      <td>Republic Bank of Chicago</td>
      <td>May 8, 2015</td>
      <td>July 12, 2016</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Doral Bank  En Espanol</td>
      <td>San Juan</td>
      <td>PR</td>
      <td>32102</td>
      <td>Banco Popular de Puerto Rico</td>
      <td>February 27, 2015</td>
      <td>May 13, 2015</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Capitol City Bank &amp; Trust Company</td>
      <td>Atlanta</td>
      <td>GA</td>
      <td>33938</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>February 13, 2015</td>
      <td>April 21, 2015</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Highland Community Bank</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>20290</td>
      <td>United Fidelity Bank, fsb</td>
      <td>January 23, 2015</td>
      <td>November 15, 2017</td>
    </tr>
    <tr>
      <th>20</th>
      <td>First National Bank of Crestview</td>
      <td>Crestview</td>
      <td>FL</td>
      <td>17557</td>
      <td>First NBC Bank</td>
      <td>January 16, 2015</td>
      <td>November 15, 2017</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Northern Star Bank</td>
      <td>Mankato</td>
      <td>MN</td>
      <td>34983</td>
      <td>BankVista</td>
      <td>December 19, 2014</td>
      <td>January 3, 2018</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Frontier Bank, FSB D/B/A El Paseo Bank</td>
      <td>Palm Desert</td>
      <td>CA</td>
      <td>34738</td>
      <td>Bank of Southern California, N.A.</td>
      <td>November 7, 2014</td>
      <td>November 10, 2016</td>
    </tr>
    <tr>
      <th>23</th>
      <td>The National Republic Bank of Chicago</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>916</td>
      <td>State Bank of Texas</td>
      <td>October 24, 2014</td>
      <td>January 6, 2016</td>
    </tr>
    <tr>
      <th>24</th>
      <td>NBRS Financial</td>
      <td>Rising Sun</td>
      <td>MD</td>
      <td>4862</td>
      <td>Howard Bank</td>
      <td>October 17, 2014</td>
      <td>February 19, 2018</td>
    </tr>
    <tr>
      <th>25</th>
      <td>GreenChoice Bank, fsb</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>28462</td>
      <td>Providence Bank, LLC</td>
      <td>July 25, 2014</td>
      <td>December 12, 2016</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Eastside Commercial Bank</td>
      <td>Conyers</td>
      <td>GA</td>
      <td>58125</td>
      <td>Community &amp; Southern Bank</td>
      <td>July 18, 2014</td>
      <td>October 6, 2017</td>
    </tr>
    <tr>
      <th>27</th>
      <td>The Freedom State Bank</td>
      <td>Freedom</td>
      <td>OK</td>
      <td>12483</td>
      <td>Alva State Bank &amp; Trust Company</td>
      <td>June 27, 2014</td>
      <td>February 21, 2018</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Valley Bank</td>
      <td>Fort Lauderdale</td>
      <td>FL</td>
      <td>21793</td>
      <td>Landmark Bank, National Association</td>
      <td>June 20, 2014</td>
      <td>February 14, 2018</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Valley Bank</td>
      <td>Moline</td>
      <td>IL</td>
      <td>10450</td>
      <td>Great Southern Bank</td>
      <td>June 20, 2014</td>
      <td>June 26, 2015</td>
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
    </tr>
    <tr>
      <th>525</th>
      <td>ANB Financial, NA</td>
      <td>Bentonville</td>
      <td>AR</td>
      <td>33901</td>
      <td>Pulaski Bank and Trust Company</td>
      <td>May 9, 2008</td>
      <td>August 28, 2012</td>
    </tr>
    <tr>
      <th>526</th>
      <td>Hume Bank</td>
      <td>Hume</td>
      <td>MO</td>
      <td>1971</td>
      <td>Security Bank</td>
      <td>March 7, 2008</td>
      <td>August 28, 2012</td>
    </tr>
    <tr>
      <th>527</th>
      <td>Douglass National Bank</td>
      <td>Kansas City</td>
      <td>MO</td>
      <td>24660</td>
      <td>Liberty Bank and Trust Company</td>
      <td>January 25, 2008</td>
      <td>October 26, 2012</td>
    </tr>
    <tr>
      <th>528</th>
      <td>Miami Valley Bank</td>
      <td>Lakeview</td>
      <td>OH</td>
      <td>16848</td>
      <td>The Citizens Banking Company</td>
      <td>October 4, 2007</td>
      <td>September 12, 2016</td>
    </tr>
    <tr>
      <th>529</th>
      <td>NetBank</td>
      <td>Alpharetta</td>
      <td>GA</td>
      <td>32575</td>
      <td>ING DIRECT</td>
      <td>September 28, 2007</td>
      <td>August 28, 2012</td>
    </tr>
    <tr>
      <th>530</th>
      <td>Metropolitan Savings Bank</td>
      <td>Pittsburgh</td>
      <td>PA</td>
      <td>35353</td>
      <td>Allegheny Valley Bank of Pittsburgh</td>
      <td>February 2, 2007</td>
      <td>October 27, 2010</td>
    </tr>
    <tr>
      <th>531</th>
      <td>Bank of Ephraim</td>
      <td>Ephraim</td>
      <td>UT</td>
      <td>1249</td>
      <td>Far West Bank</td>
      <td>June 25, 2004</td>
      <td>April 9, 2008</td>
    </tr>
    <tr>
      <th>532</th>
      <td>Reliance Bank</td>
      <td>White Plains</td>
      <td>NY</td>
      <td>26778</td>
      <td>Union State Bank</td>
      <td>March 19, 2004</td>
      <td>April 9, 2008</td>
    </tr>
    <tr>
      <th>533</th>
      <td>Guaranty National Bank of Tallahassee</td>
      <td>Tallahassee</td>
      <td>FL</td>
      <td>26838</td>
      <td>Hancock Bank of Florida</td>
      <td>March 12, 2004</td>
      <td>April 17, 2018</td>
    </tr>
    <tr>
      <th>534</th>
      <td>Dollar Savings Bank</td>
      <td>Newark</td>
      <td>NJ</td>
      <td>31330</td>
      <td>No Acquirer</td>
      <td>February 14, 2004</td>
      <td>April 9, 2008</td>
    </tr>
    <tr>
      <th>535</th>
      <td>Pulaski Savings Bank</td>
      <td>Philadelphia</td>
      <td>PA</td>
      <td>27203</td>
      <td>Earthstar Bank</td>
      <td>November 14, 2003</td>
      <td>October 6, 2017</td>
    </tr>
    <tr>
      <th>536</th>
      <td>First National Bank of Blanchardville</td>
      <td>Blanchardville</td>
      <td>WI</td>
      <td>11639</td>
      <td>The Park Bank</td>
      <td>May 9, 2003</td>
      <td>June 5, 2012</td>
    </tr>
    <tr>
      <th>537</th>
      <td>Southern Pacific Bank</td>
      <td>Torrance</td>
      <td>CA</td>
      <td>27094</td>
      <td>Beal Bank</td>
      <td>February 7, 2003</td>
      <td>October 20, 2008</td>
    </tr>
    <tr>
      <th>538</th>
      <td>Farmers Bank of Cheneyville</td>
      <td>Cheneyville</td>
      <td>LA</td>
      <td>16445</td>
      <td>Sabine State Bank &amp; Trust</td>
      <td>December 17, 2002</td>
      <td>October 20, 2004</td>
    </tr>
    <tr>
      <th>539</th>
      <td>Bank of Alamo</td>
      <td>Alamo</td>
      <td>TN</td>
      <td>9961</td>
      <td>No Acquirer</td>
      <td>November 8, 2002</td>
      <td>March 18, 2005</td>
    </tr>
    <tr>
      <th>540</th>
      <td>AmTrade International Bank  En Espanol</td>
      <td>Atlanta</td>
      <td>GA</td>
      <td>33784</td>
      <td>No Acquirer</td>
      <td>September 30, 2002</td>
      <td>September 11, 2006</td>
    </tr>
    <tr>
      <th>541</th>
      <td>Universal Federal Savings Bank</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>29355</td>
      <td>Chicago Community Bank</td>
      <td>June 27, 2002</td>
      <td>October 6, 2017</td>
    </tr>
    <tr>
      <th>542</th>
      <td>Connecticut Bank of Commerce</td>
      <td>Stamford</td>
      <td>CT</td>
      <td>19183</td>
      <td>Hudson United Bank</td>
      <td>June 26, 2002</td>
      <td>February 14, 2012</td>
    </tr>
    <tr>
      <th>543</th>
      <td>New Century Bank</td>
      <td>Shelby Township</td>
      <td>MI</td>
      <td>34979</td>
      <td>No Acquirer</td>
      <td>March 28, 2002</td>
      <td>March 18, 2005</td>
    </tr>
    <tr>
      <th>544</th>
      <td>Net 1st National Bank</td>
      <td>Boca Raton</td>
      <td>FL</td>
      <td>26652</td>
      <td>Bank Leumi USA</td>
      <td>March 1, 2002</td>
      <td>April 9, 2008</td>
    </tr>
    <tr>
      <th>545</th>
      <td>NextBank, NA</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>22314</td>
      <td>No Acquirer</td>
      <td>February 7, 2002</td>
      <td>February 5, 2015</td>
    </tr>
    <tr>
      <th>546</th>
      <td>Oakwood Deposit Bank Co.</td>
      <td>Oakwood</td>
      <td>OH</td>
      <td>8966</td>
      <td>The State Bank &amp; Trust Company</td>
      <td>February 1, 2002</td>
      <td>October 25, 2012</td>
    </tr>
    <tr>
      <th>547</th>
      <td>Bank of Sierra Blanca</td>
      <td>Sierra Blanca</td>
      <td>TX</td>
      <td>22002</td>
      <td>The Security State Bank of Pecos</td>
      <td>January 18, 2002</td>
      <td>November 6, 2003</td>
    </tr>
    <tr>
      <th>548</th>
      <td>Hamilton Bank, NA  En Espanol</td>
      <td>Miami</td>
      <td>FL</td>
      <td>24382</td>
      <td>Israel Discount Bank of New York</td>
      <td>January 11, 2002</td>
      <td>September 21, 2015</td>
    </tr>
    <tr>
      <th>549</th>
      <td>Sinclair National Bank</td>
      <td>Gravette</td>
      <td>AR</td>
      <td>34248</td>
      <td>Delta Trust &amp; Bank</td>
      <td>September 7, 2001</td>
      <td>October 6, 2017</td>
    </tr>
    <tr>
      <th>550</th>
      <td>Superior Bank, FSB</td>
      <td>Hinsdale</td>
      <td>IL</td>
      <td>32646</td>
      <td>Superior Federal, FSB</td>
      <td>July 27, 2001</td>
      <td>August 19, 2014</td>
    </tr>
    <tr>
      <th>551</th>
      <td>Malta National Bank</td>
      <td>Malta</td>
      <td>OH</td>
      <td>6629</td>
      <td>North Valley Bank</td>
      <td>May 3, 2001</td>
      <td>November 18, 2002</td>
    </tr>
    <tr>
      <th>552</th>
      <td>First Alliance Bank &amp; Trust Co.</td>
      <td>Manchester</td>
      <td>NH</td>
      <td>34264</td>
      <td>Southern New Hampshire Bank &amp; Trust</td>
      <td>February 2, 2001</td>
      <td>February 18, 2003</td>
    </tr>
    <tr>
      <th>553</th>
      <td>National State Bank of Metropolis</td>
      <td>Metropolis</td>
      <td>IL</td>
      <td>3815</td>
      <td>Banterra Bank of Marion</td>
      <td>December 14, 2000</td>
      <td>March 17, 2005</td>
    </tr>
    <tr>
      <th>554</th>
      <td>Bank of Honolulu</td>
      <td>Honolulu</td>
      <td>HI</td>
      <td>21029</td>
      <td>Bank of the Orient</td>
      <td>October 13, 2000</td>
      <td>March 17, 2005</td>
    </tr>
  </tbody>
</table>
<p>555 rows × 7 columns</p>
</div>




```python
## Reading from SQL depends on which flavor of SQL is being used. The example below is for creating a sqllite engine
## and writing/reading to/from it.
```


```python
from sqlalchemy import create_engine
```


```python
engine = create_engine('sqlite:///:memory:') #sql db running in memory
```


```python
df6.to_sql('sqltable',engine)
```


```python
sqldf = pd.read_sql('sqltable',con=engine)
sqldf
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
      <th>index</th>
      <th>Bank Name</th>
      <th>City</th>
      <th>ST</th>
      <th>CERT</th>
      <th>Acquiring Institution</th>
      <th>Closing Date</th>
      <th>Updated Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Washington Federal Bank for Savings</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>30570</td>
      <td>Royal Savings Bank</td>
      <td>December 15, 2017</td>
      <td>February 21, 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>The Farmers and Merchants State Bank of Argonia</td>
      <td>Argonia</td>
      <td>KS</td>
      <td>17719</td>
      <td>Conway Bank</td>
      <td>October 13, 2017</td>
      <td>February 21, 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Fayette County Bank</td>
      <td>Saint Elmo</td>
      <td>IL</td>
      <td>1802</td>
      <td>United Fidelity Bank, fsb</td>
      <td>May 26, 2017</td>
      <td>July 26, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Guaranty Bank, (d/b/a BestBank in Georgia &amp; Mi...</td>
      <td>Milwaukee</td>
      <td>WI</td>
      <td>30003</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>May 5, 2017</td>
      <td>March 22, 2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>First NBC Bank</td>
      <td>New Orleans</td>
      <td>LA</td>
      <td>58302</td>
      <td>Whitney Bank</td>
      <td>April 28, 2017</td>
      <td>December 5, 2017</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Proficio Bank</td>
      <td>Cottonwood Heights</td>
      <td>UT</td>
      <td>35495</td>
      <td>Cache Valley Bank</td>
      <td>March 3, 2017</td>
      <td>March 7, 2018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Seaway Bank and Trust Company</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>19328</td>
      <td>State Bank of Texas</td>
      <td>January 27, 2017</td>
      <td>May 18, 2017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Harvest Community Bank</td>
      <td>Pennsville</td>
      <td>NJ</td>
      <td>34951</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>January 13, 2017</td>
      <td>May 18, 2017</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Allied Bank</td>
      <td>Mulberry</td>
      <td>AR</td>
      <td>91</td>
      <td>Today's Bank</td>
      <td>September 23, 2016</td>
      <td>September 25, 2017</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>The Woodbury Banking Company</td>
      <td>Woodbury</td>
      <td>GA</td>
      <td>11297</td>
      <td>United Bank</td>
      <td>August 19, 2016</td>
      <td>June 1, 2017</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>First CornerStone Bank</td>
      <td>King of Prussia</td>
      <td>PA</td>
      <td>35312</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>May 6, 2016</td>
      <td>September 6, 2016</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>Trust Company Bank</td>
      <td>Memphis</td>
      <td>TN</td>
      <td>9956</td>
      <td>The Bank of Fayette County</td>
      <td>April 29, 2016</td>
      <td>September 6, 2016</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>North Milwaukee State Bank</td>
      <td>Milwaukee</td>
      <td>WI</td>
      <td>20364</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>March 11, 2016</td>
      <td>March 13, 2017</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>Hometown National Bank</td>
      <td>Longview</td>
      <td>WA</td>
      <td>35156</td>
      <td>Twin City Bank</td>
      <td>October 2, 2015</td>
      <td>February 19, 2018</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>The Bank of Georgia</td>
      <td>Peachtree City</td>
      <td>GA</td>
      <td>35259</td>
      <td>Fidelity Bank</td>
      <td>October 2, 2015</td>
      <td>July 9, 2018</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>Premier Bank</td>
      <td>Denver</td>
      <td>CO</td>
      <td>34112</td>
      <td>United Fidelity Bank, fsb</td>
      <td>July 10, 2015</td>
      <td>February 20, 2018</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>Edgebrook Bank</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>57772</td>
      <td>Republic Bank of Chicago</td>
      <td>May 8, 2015</td>
      <td>July 12, 2016</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>Doral Bank  En Espanol</td>
      <td>San Juan</td>
      <td>PR</td>
      <td>32102</td>
      <td>Banco Popular de Puerto Rico</td>
      <td>February 27, 2015</td>
      <td>May 13, 2015</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>Capitol City Bank &amp; Trust Company</td>
      <td>Atlanta</td>
      <td>GA</td>
      <td>33938</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>February 13, 2015</td>
      <td>April 21, 2015</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>Highland Community Bank</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>20290</td>
      <td>United Fidelity Bank, fsb</td>
      <td>January 23, 2015</td>
      <td>November 15, 2017</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>First National Bank of Crestview</td>
      <td>Crestview</td>
      <td>FL</td>
      <td>17557</td>
      <td>First NBC Bank</td>
      <td>January 16, 2015</td>
      <td>November 15, 2017</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>Northern Star Bank</td>
      <td>Mankato</td>
      <td>MN</td>
      <td>34983</td>
      <td>BankVista</td>
      <td>December 19, 2014</td>
      <td>January 3, 2018</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>Frontier Bank, FSB D/B/A El Paseo Bank</td>
      <td>Palm Desert</td>
      <td>CA</td>
      <td>34738</td>
      <td>Bank of Southern California, N.A.</td>
      <td>November 7, 2014</td>
      <td>November 10, 2016</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>The National Republic Bank of Chicago</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>916</td>
      <td>State Bank of Texas</td>
      <td>October 24, 2014</td>
      <td>January 6, 2016</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>NBRS Financial</td>
      <td>Rising Sun</td>
      <td>MD</td>
      <td>4862</td>
      <td>Howard Bank</td>
      <td>October 17, 2014</td>
      <td>February 19, 2018</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>GreenChoice Bank, fsb</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>28462</td>
      <td>Providence Bank, LLC</td>
      <td>July 25, 2014</td>
      <td>December 12, 2016</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>Eastside Commercial Bank</td>
      <td>Conyers</td>
      <td>GA</td>
      <td>58125</td>
      <td>Community &amp; Southern Bank</td>
      <td>July 18, 2014</td>
      <td>October 6, 2017</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>The Freedom State Bank</td>
      <td>Freedom</td>
      <td>OK</td>
      <td>12483</td>
      <td>Alva State Bank &amp; Trust Company</td>
      <td>June 27, 2014</td>
      <td>February 21, 2018</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>Valley Bank</td>
      <td>Fort Lauderdale</td>
      <td>FL</td>
      <td>21793</td>
      <td>Landmark Bank, National Association</td>
      <td>June 20, 2014</td>
      <td>February 14, 2018</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>Valley Bank</td>
      <td>Moline</td>
      <td>IL</td>
      <td>10450</td>
      <td>Great Southern Bank</td>
      <td>June 20, 2014</td>
      <td>June 26, 2015</td>
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
    </tr>
    <tr>
      <th>525</th>
      <td>525</td>
      <td>ANB Financial, NA</td>
      <td>Bentonville</td>
      <td>AR</td>
      <td>33901</td>
      <td>Pulaski Bank and Trust Company</td>
      <td>May 9, 2008</td>
      <td>August 28, 2012</td>
    </tr>
    <tr>
      <th>526</th>
      <td>526</td>
      <td>Hume Bank</td>
      <td>Hume</td>
      <td>MO</td>
      <td>1971</td>
      <td>Security Bank</td>
      <td>March 7, 2008</td>
      <td>August 28, 2012</td>
    </tr>
    <tr>
      <th>527</th>
      <td>527</td>
      <td>Douglass National Bank</td>
      <td>Kansas City</td>
      <td>MO</td>
      <td>24660</td>
      <td>Liberty Bank and Trust Company</td>
      <td>January 25, 2008</td>
      <td>October 26, 2012</td>
    </tr>
    <tr>
      <th>528</th>
      <td>528</td>
      <td>Miami Valley Bank</td>
      <td>Lakeview</td>
      <td>OH</td>
      <td>16848</td>
      <td>The Citizens Banking Company</td>
      <td>October 4, 2007</td>
      <td>September 12, 2016</td>
    </tr>
    <tr>
      <th>529</th>
      <td>529</td>
      <td>NetBank</td>
      <td>Alpharetta</td>
      <td>GA</td>
      <td>32575</td>
      <td>ING DIRECT</td>
      <td>September 28, 2007</td>
      <td>August 28, 2012</td>
    </tr>
    <tr>
      <th>530</th>
      <td>530</td>
      <td>Metropolitan Savings Bank</td>
      <td>Pittsburgh</td>
      <td>PA</td>
      <td>35353</td>
      <td>Allegheny Valley Bank of Pittsburgh</td>
      <td>February 2, 2007</td>
      <td>October 27, 2010</td>
    </tr>
    <tr>
      <th>531</th>
      <td>531</td>
      <td>Bank of Ephraim</td>
      <td>Ephraim</td>
      <td>UT</td>
      <td>1249</td>
      <td>Far West Bank</td>
      <td>June 25, 2004</td>
      <td>April 9, 2008</td>
    </tr>
    <tr>
      <th>532</th>
      <td>532</td>
      <td>Reliance Bank</td>
      <td>White Plains</td>
      <td>NY</td>
      <td>26778</td>
      <td>Union State Bank</td>
      <td>March 19, 2004</td>
      <td>April 9, 2008</td>
    </tr>
    <tr>
      <th>533</th>
      <td>533</td>
      <td>Guaranty National Bank of Tallahassee</td>
      <td>Tallahassee</td>
      <td>FL</td>
      <td>26838</td>
      <td>Hancock Bank of Florida</td>
      <td>March 12, 2004</td>
      <td>April 17, 2018</td>
    </tr>
    <tr>
      <th>534</th>
      <td>534</td>
      <td>Dollar Savings Bank</td>
      <td>Newark</td>
      <td>NJ</td>
      <td>31330</td>
      <td>No Acquirer</td>
      <td>February 14, 2004</td>
      <td>April 9, 2008</td>
    </tr>
    <tr>
      <th>535</th>
      <td>535</td>
      <td>Pulaski Savings Bank</td>
      <td>Philadelphia</td>
      <td>PA</td>
      <td>27203</td>
      <td>Earthstar Bank</td>
      <td>November 14, 2003</td>
      <td>October 6, 2017</td>
    </tr>
    <tr>
      <th>536</th>
      <td>536</td>
      <td>First National Bank of Blanchardville</td>
      <td>Blanchardville</td>
      <td>WI</td>
      <td>11639</td>
      <td>The Park Bank</td>
      <td>May 9, 2003</td>
      <td>June 5, 2012</td>
    </tr>
    <tr>
      <th>537</th>
      <td>537</td>
      <td>Southern Pacific Bank</td>
      <td>Torrance</td>
      <td>CA</td>
      <td>27094</td>
      <td>Beal Bank</td>
      <td>February 7, 2003</td>
      <td>October 20, 2008</td>
    </tr>
    <tr>
      <th>538</th>
      <td>538</td>
      <td>Farmers Bank of Cheneyville</td>
      <td>Cheneyville</td>
      <td>LA</td>
      <td>16445</td>
      <td>Sabine State Bank &amp; Trust</td>
      <td>December 17, 2002</td>
      <td>October 20, 2004</td>
    </tr>
    <tr>
      <th>539</th>
      <td>539</td>
      <td>Bank of Alamo</td>
      <td>Alamo</td>
      <td>TN</td>
      <td>9961</td>
      <td>No Acquirer</td>
      <td>November 8, 2002</td>
      <td>March 18, 2005</td>
    </tr>
    <tr>
      <th>540</th>
      <td>540</td>
      <td>AmTrade International Bank  En Espanol</td>
      <td>Atlanta</td>
      <td>GA</td>
      <td>33784</td>
      <td>No Acquirer</td>
      <td>September 30, 2002</td>
      <td>September 11, 2006</td>
    </tr>
    <tr>
      <th>541</th>
      <td>541</td>
      <td>Universal Federal Savings Bank</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>29355</td>
      <td>Chicago Community Bank</td>
      <td>June 27, 2002</td>
      <td>October 6, 2017</td>
    </tr>
    <tr>
      <th>542</th>
      <td>542</td>
      <td>Connecticut Bank of Commerce</td>
      <td>Stamford</td>
      <td>CT</td>
      <td>19183</td>
      <td>Hudson United Bank</td>
      <td>June 26, 2002</td>
      <td>February 14, 2012</td>
    </tr>
    <tr>
      <th>543</th>
      <td>543</td>
      <td>New Century Bank</td>
      <td>Shelby Township</td>
      <td>MI</td>
      <td>34979</td>
      <td>No Acquirer</td>
      <td>March 28, 2002</td>
      <td>March 18, 2005</td>
    </tr>
    <tr>
      <th>544</th>
      <td>544</td>
      <td>Net 1st National Bank</td>
      <td>Boca Raton</td>
      <td>FL</td>
      <td>26652</td>
      <td>Bank Leumi USA</td>
      <td>March 1, 2002</td>
      <td>April 9, 2008</td>
    </tr>
    <tr>
      <th>545</th>
      <td>545</td>
      <td>NextBank, NA</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>22314</td>
      <td>No Acquirer</td>
      <td>February 7, 2002</td>
      <td>February 5, 2015</td>
    </tr>
    <tr>
      <th>546</th>
      <td>546</td>
      <td>Oakwood Deposit Bank Co.</td>
      <td>Oakwood</td>
      <td>OH</td>
      <td>8966</td>
      <td>The State Bank &amp; Trust Company</td>
      <td>February 1, 2002</td>
      <td>October 25, 2012</td>
    </tr>
    <tr>
      <th>547</th>
      <td>547</td>
      <td>Bank of Sierra Blanca</td>
      <td>Sierra Blanca</td>
      <td>TX</td>
      <td>22002</td>
      <td>The Security State Bank of Pecos</td>
      <td>January 18, 2002</td>
      <td>November 6, 2003</td>
    </tr>
    <tr>
      <th>548</th>
      <td>548</td>
      <td>Hamilton Bank, NA  En Espanol</td>
      <td>Miami</td>
      <td>FL</td>
      <td>24382</td>
      <td>Israel Discount Bank of New York</td>
      <td>January 11, 2002</td>
      <td>September 21, 2015</td>
    </tr>
    <tr>
      <th>549</th>
      <td>549</td>
      <td>Sinclair National Bank</td>
      <td>Gravette</td>
      <td>AR</td>
      <td>34248</td>
      <td>Delta Trust &amp; Bank</td>
      <td>September 7, 2001</td>
      <td>October 6, 2017</td>
    </tr>
    <tr>
      <th>550</th>
      <td>550</td>
      <td>Superior Bank, FSB</td>
      <td>Hinsdale</td>
      <td>IL</td>
      <td>32646</td>
      <td>Superior Federal, FSB</td>
      <td>July 27, 2001</td>
      <td>August 19, 2014</td>
    </tr>
    <tr>
      <th>551</th>
      <td>551</td>
      <td>Malta National Bank</td>
      <td>Malta</td>
      <td>OH</td>
      <td>6629</td>
      <td>North Valley Bank</td>
      <td>May 3, 2001</td>
      <td>November 18, 2002</td>
    </tr>
    <tr>
      <th>552</th>
      <td>552</td>
      <td>First Alliance Bank &amp; Trust Co.</td>
      <td>Manchester</td>
      <td>NH</td>
      <td>34264</td>
      <td>Southern New Hampshire Bank &amp; Trust</td>
      <td>February 2, 2001</td>
      <td>February 18, 2003</td>
    </tr>
    <tr>
      <th>553</th>
      <td>553</td>
      <td>National State Bank of Metropolis</td>
      <td>Metropolis</td>
      <td>IL</td>
      <td>3815</td>
      <td>Banterra Bank of Marion</td>
      <td>December 14, 2000</td>
      <td>March 17, 2005</td>
    </tr>
    <tr>
      <th>554</th>
      <td>554</td>
      <td>Bank of Honolulu</td>
      <td>Honolulu</td>
      <td>HI</td>
      <td>21029</td>
      <td>Bank of the Orient</td>
      <td>October 13, 2000</td>
      <td>March 17, 2005</td>
    </tr>
  </tbody>
</table>
<p>555 rows × 8 columns</p>
</div>




```python

```
