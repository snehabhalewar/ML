{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "38894932-405e-4ddf-a652-23ac01a3cb2a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "df=pd.read_csv(\"Wine.csv\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "80792369-6e8c-45ea-b9f2-8d3facd37bad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Alcohol  Malic_Acid   Ash  Ash_Alcanity  Magnesium  Total_Phenols  \\\n",
      "0    14.23        1.71  2.43          15.6        127           2.80   \n",
      "1    13.20        1.78  2.14          11.2        100           2.65   \n",
      "2    13.16        2.36  2.67          18.6        101           2.80   \n",
      "3    14.37        1.95  2.50          16.8        113           3.85   \n",
      "4    13.24        2.59  2.87          21.0        118           2.80   \n",
      "\n",
      "   Flavanoids  Nonflavanoid_Phenols  Proanthocyanins  Color_Intensity   Hue  \\\n",
      "0        3.06                  0.28             2.29             5.64  1.04   \n",
      "1        2.76                  0.26             1.28             4.38  1.05   \n",
      "2        3.24                  0.30             2.81             5.68  1.03   \n",
      "3        3.49                  0.24             2.18             7.80  0.86   \n",
      "4        2.69                  0.39             1.82             4.32  1.04   \n",
      "\n",
      "   OD280  Proline  Customer_Segment  \n",
      "0   3.92     1065                 1  \n",
      "1   3.40     1050                 1  \n",
      "2   3.17     1185                 1  \n",
      "3   3.45     1480                 1  \n",
      "4   2.93      735                 1  \n"
     ]
    }
   ],
   "source": [
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "3d06c823-157f-44e2-8b1f-f597a5942ba0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "          Alcohol  Malic_Acid         Ash  Ash_Alcanity   Magnesium  \\\n",
      "count  178.000000  178.000000  178.000000    178.000000  178.000000   \n",
      "mean    13.000618    2.336348    2.366517     19.494944   99.741573   \n",
      "std      0.811827    1.117146    0.274344      3.339564   14.282484   \n",
      "min     11.030000    0.740000    1.360000     10.600000   70.000000   \n",
      "25%     12.362500    1.602500    2.210000     17.200000   88.000000   \n",
      "50%     13.050000    1.865000    2.360000     19.500000   98.000000   \n",
      "75%     13.677500    3.082500    2.557500     21.500000  107.000000   \n",
      "max     14.830000    5.800000    3.230000     30.000000  162.000000   \n",
      "\n",
      "       Total_Phenols  Flavanoids  Nonflavanoid_Phenols  Proanthocyanins  \\\n",
      "count     178.000000  178.000000            178.000000       178.000000   \n",
      "mean        2.295112    2.029270              0.361854         1.590899   \n",
      "std         0.625851    0.998859              0.124453         0.572359   \n",
      "min         0.980000    0.340000              0.130000         0.410000   \n",
      "25%         1.742500    1.205000              0.270000         1.250000   \n",
      "50%         2.355000    2.135000              0.340000         1.555000   \n",
      "75%         2.800000    2.875000              0.437500         1.950000   \n",
      "max         3.880000    5.080000              0.660000         3.580000   \n",
      "\n",
      "       Color_Intensity         Hue       OD280      Proline  Customer_Segment  \n",
      "count       178.000000  178.000000  178.000000   178.000000        178.000000  \n",
      "mean          5.058090    0.957449    2.611685   746.893258          1.938202  \n",
      "std           2.318286    0.228572    0.709990   314.907474          0.775035  \n",
      "min           1.280000    0.480000    1.270000   278.000000          1.000000  \n",
      "25%           3.220000    0.782500    1.937500   500.500000          1.000000  \n",
      "50%           4.690000    0.965000    2.780000   673.500000          2.000000  \n",
      "75%           6.200000    1.120000    3.170000   985.000000          3.000000  \n",
      "max          13.000000    1.710000    4.000000  1680.000000          3.000000  \n"
     ]
    }
   ],
   "source": [
    "print(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "f7345337-8cbc-4c20-9478-e406de3c0a1a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 178 entries, 0 to 177\n",
      "Data columns (total 14 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   Alcohol               178 non-null    float64\n",
      " 1   Malic_Acid            178 non-null    float64\n",
      " 2   Ash                   178 non-null    float64\n",
      " 3   Ash_Alcanity          178 non-null    float64\n",
      " 4   Magnesium             178 non-null    int64  \n",
      " 5   Total_Phenols         178 non-null    float64\n",
      " 6   Flavanoids            178 non-null    float64\n",
      " 7   Nonflavanoid_Phenols  178 non-null    float64\n",
      " 8   Proanthocyanins       178 non-null    float64\n",
      " 9   Color_Intensity       178 non-null    float64\n",
      " 10  Hue                   178 non-null    float64\n",
      " 11  OD280                 178 non-null    float64\n",
      " 12  Proline               178 non-null    int64  \n",
      " 13  Customer_Segment      178 non-null    int64  \n",
      "dtypes: float64(11), int64(3)\n",
      "memory usage: 19.6 KB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "643b060e-4ea2-44dd-8613-9edbf4b6852c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(178, 14)\n"
     ]
    }
   ],
   "source": [
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "48bff2c9-8964-4e15-95fa-1e7e97bcf279",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline', 'Customer_Segment']\n"
     ]
    }
   ],
   "source": [
    "print(df.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "f0abfd54-a18a-4709-b245-859b535fbc8e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Alcohol                 0\n",
      "Malic_Acid              0\n",
      "Ash                     0\n",
      "Ash_Alcanity            0\n",
      "Magnesium               0\n",
      "Total_Phenols           0\n",
      "Flavanoids              0\n",
      "Nonflavanoid_Phenols    0\n",
      "Proanthocyanins         0\n",
      "Color_Intensity         0\n",
      "Hue                     0\n",
      "OD280                   0\n",
      "Proline                 0\n",
      "Customer_Segment        0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "783f506a-e45a-412c-aaa5-03622e9e9f52",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df.drop('Customer_Segment', axis=1)\n",
    "y=df['Customer_Segment']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "baac370a-f537-4e22-904e-1af8e1dd57b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler=StandardScaler()\n",
    "x_scaled=scaler.fit_transform(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "3bdd1453-e9a4-4344-a125-67f34e554816",
   "metadata": {},
   "outputs": [],
   "source": [
    "pca=PCA(n_components=2)\n",
    "x_pca=pca.fit_transform(x_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "9fda5346-949f-440d-b3ea-31f33914f10a",
   "metadata": {},
   "outputs": [],
   "source": [
    "pca_df=pd.DataFrame(x_pca, columns=['Principal Component1','Principal Component2'])\n",
    "pca_df['Customer_Segment']=y.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "ea972c5d-35c4-4eeb-9328-4e694b30c5a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Principal Component1  Principal Component2  Customer_Segment\n",
      "0              3.316751              1.443463                 1\n",
      "1              2.209465             -0.333393                 1\n",
      "2              2.516740              1.031151                 1\n",
      "3              3.757066              2.756372                 1\n",
      "4              1.008908              0.869831                 1\n"
     ]
    }
   ],
   "source": [
    "print(pca_df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "ff053a46-d8fe-4896-b166-22273c0fef35",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.36198848 0.1920749 ]\n"
     ]
    }
   ],
   "source": [
    "print(pca.explained_variance_ratio_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a5c1234f-cf61-40e6-8b48-b05930517cf3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "55.41 %\n"
     ]
    }
   ],
   "source": [
    "print(round(sum(pca.explained_variance_ratio_)*100,2),'%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "053239f0-b9f7-40b3-b72a-fe610193722e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sneha\\AppData\\Local\\Temp\\ipykernel_6836\\145480253.py:3: UserWarning: \n",
      "The palette list has fewer values (2) than needed (3) and will cycle, which may produce an uninterpretable plot.\n",
      "  sns.scatterplot(\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA00AAAJvCAYAAACnEDqlAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvctoD+AAAAAlwSFlzAAAPYQAAD2EBqD+naQAAuTtJREFUeJzs3Qm8zPX3x/Eje/aIyhLJHklIKZHIUmmTpdKqTSqltJe0UNo3pX5ataC0KkQLoo0WkTVky77v7v/x/nz/c82dOzN3nf31fDzmwZ3vvXO/s9yZ7/me8zmnQFpaWpoBAAAAAII6KPjVAAAAAACCJgAAAADIApkmAAAAAAiDoAkAAAAAwiBoAgAAAIAwCJoAAAAAIAyCJgAAAAAIg6AJAAAAAMIgaAIARMXevXvt8ccft9atW1u9evXs3nvvjdjvGjFihNWtW9e2bNkSsd8BAEgdBE0AEOd2797tAgDfRQFHs2bN7Morr7Sff/456M98/vnndsUVV9iJJ55oxx13nJ111ll2xx132Lx584J+//bt261x48bu9mfNmhWR+/HUU0/ZfffdZ/369bOPPvrI+vTpk+l7duzYYQ0aNLCbb74507YvvvjC7d+ll16aadsPP/zgtr355pvu63Xr1tnff/9t+/bts2gbPXp0hufr2GOPtVNPPdV69+5tI0eOdPcxt4YNG+ZuU89XPIi3/QGASCFoAoA4t3//fhcA1K5d28aOHWsffvihPfjggzZ79mxr0aKFCyZ8Nm7caO3bt7fu3bvbYYcdZoMHD7bXX3/dLrvsMps5c6bVqVPHBRiBRo0aZb/99putWLHCXnnllYjcj08//dQFcV26dHEH2tq/QMWLF7eDDz7Y3n///aCB4MKFC11QokDS31dffeUeo2OOOcZ9rYBxzpw5Vrp0aYs2PQfal/79+7vn65133rGBAwfaUUcdZQMGDHDPwffff5+r2167dq27bb0m4kG87Q8ARApBEwAkCAUAvkxTx44dXRBSqFAhu//++9O/p2fPnjZ16lT77rvv7JFHHnEZDmU6zj//fBdYPPPMM7Znz55Mtz18+HD3vdddd507yI9E5mDVqlXZCmJOO+00970KevxNmjTJevXq5fZt+vTpmbaVK1fOZcvkkEMOcY/VQQfF7mOuSpUqbh8UyKkk8c4777Q//vjDDj30UOvcubMtXrw4ZvsGAMgZgiYASFA6+Fb2wldyN3nyZBs3bpzdcsstriQvmBtvvNFatmyZ4ToFJwq0FDBde+21tnXrVvvggw9ytFbphRdesHbt2lmjRo3cvy+99JK7Xr7++msXPPzzzz/p/9dFmbJQQZPv/visXLnS5s6d6zJo+lkFST4qd5sxY4YLTHxBUrA1TQoYdZ3268UXX3SPg8oc7777btu5c2em/di0aZM9/PDD7nZ1v84880wbM2aM5UXZsmXt+eefd/ul9V0+CoD9yy+bNm1ql19+uf3yyy/p36M1YLoPcvzxx2d6HLNzG7Jt2zZ77LHH3POk18m5557ryhoDs0VZ3f+s9gcAkkmhWO8AACB3tF5n9erVVr58eff1xx9/7P7VQXA4BQsWzPD1q6++6krlzjvvPCtcuLCdccYZ7jqV9GVFB9oqt/v2229dKWDz5s1dAKPStC+//NKtXTrhhBNcmZput1atWi5okOrVqwe9zVNOOcWKFCniAqPrr7/eXaf/a98U6OggXl8/8MADbtuUKVNcuV7btm3TbyPYmqY1a9a46/RzxYoVc2usfv/9d7vpppts/fr1LtDz0eN68sknuyBMpZA1a9Z02buLL77YBW8KtHJLJYoKeJX581GWT4+RpKWlpZdJ6v5q3ZqyVVoDpmBLgcq7777ryhj9H8fs3IZcdNFFrhRTQZuej2XLlrmSzw0bNrjHIrv3P6v9AYCkkgYAiGs7duxI09v1RRddlH7d3r170wYMGOCuv++++9x1nTt3dl9v2rQp27e9a9eutAoVKqTdc8896dd9+umn7nZmz56d5c+/88477nvfeOONDNePGDHCXf/uu++mX1ezZs20Ll26ZGu/TjnllLTy5cun7d+/33195ZVXpp188snu/++9915akSJF0rZt2+a+vvPOO93v+uuvv9J//vHHH3fXbdiwIf26u+++21330EMPZfhdt956q7s9/8etZ8+eaaVLl05bvnx5hu994okn0goVKpS2dOnSkPs+fPhw93vGjRsX8nuaNm2aVrhw4bCPge57vXr10i677LL06wYNGuRue8uWLWF/NtRt7Ny5M61gwYJpQ4YMyfS9u3fvzvH9z+n+AECiojwPABKEr/xKjQTKlCnj1iHpjL860olvrZIyMtmlTJAyDNdcc036dZ06dXLZAmWbsqLmDMoKaS2VP2UktB+fffaZ5YZK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RWAg5e/OO++0W7QQ1S/TVLVqVWvfvr0LuJI5atYLvl27dlZY7cWRNHbu3Olex8r2hnvtxxNej4gnvB7j0P79ZmlpZinY1Cnk61Hd+15/3ezNN8127zY75xxvDpeGydIKHdF+PSYxXxVaRIKmSy+91K0n0rymaDWC0O8rWbKkzZ071zp06JB+vb6uH2ZQXtGiRd0lkF4IqfBiSJX7mQpUZrpw4UJ7/PHHbdq0aValShW7++677dhjj7XyCTJbgtcj4gmvxziwZo3Z33+bvfyyTveq05RZkyZel70Uk+H1+O+/ZjrW0WPj88IL3hDeH3/0BtZHm4K4OXPMtCRCv//kk71mGhxjJKVUen8snM37maug6ZdffrEFCxZY5cqVLVrU/OH888+3119/3a699lorVqyY/f77724904cffhi1/QBi5eeff7Y2bdqk197OmzfPJk2aZA8//LD17ds3YbJOAOD895/Zbbd5mRQffZ63aGE2erRZFI8x4s4332QMmHzWrTN7/nmvnXs0B8yqgqhdO7MFCw5cV6yY2ZdfmrVsGT9t5IF4G26rYCkWc5E0G2rHjh2uNLBXr1522mmnuWxXly5dor4vQDStWrXKrrjiiqCLFe+991439BkAEsrs2RkDJp/p072gSeV6qUgZt7feCr19zBgveIqWTZvMbropY8AkO3eade7sZcWAFHBQbsvzbrrpJjfkNpoOO+wwmzVrlmvscOKJJ7pW42+88UZU9wGIhfXr17vMUjD79+93fxcAkDC0RkflZqG8+KKXiUpFOildokTo7epe/P+jV6JWQvnpp8G3bdvmlewBKSBX+dTnn3/e/vvvPxs9erRVqFAhUyMInRWPFC1+P0/tQYEU4ptNFkohSiMAJJJ9+8y2bg2fbVFziFSkddg33GD20UfBt2ub2qVHi8a2hMv6KagCUkCugqYnnngi//cEQEiHHHKINWzY0P7444+gAVMjDWkEgEShbEmvXmZffRV8+/nn643PUtYxx5hdcknmMr0TTzQ799zods8rU8asUiWzUGXgjRtHb1+ARAuatI4IQPRUrFjR/ve//1mrVq3cuj5/zzzzjFXSBxoAJJJWrczq1lUb3IzXqxuoWmsH6XybMpRJGjrU7JprzIYNM9P7/hVXeAFKtDsL6vdpXxTEBerUKSU7HSI15bndycqVKy0tLc2O4I8GiKjG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am6954QJEzL97HPPPWeVKlWyZKOhtrogMWmmlhqYFCpUyA477LBY7w4AJOaapquuusp1e1LbXA211WXp0qXuwEDbACAVqLHD33//bZdeeqk7WaTStGuvvdadWPJXsWJFe/311+3+++93zRJEA181z8i/3BmItc2bN9vkyZOtY8eO7vWstcp67UZ63Z1KVP/55x/78MMP7ZlnnnHrANVAJdfWrDGbMcPsttvM7rzT7NdfFQnm5y4DSDGFctsIYtGiRRnOPh1++OH28ssvuzdZAEgFOmF0wgkn2KZNm9KDqLfeestllNQcx7+V+BFHHOHW+fTu3du15lY2qkKFCjHceyAjvX6//PJL69atW/p1OgFw+eWX24033mgPPviglSlTJiIB048//uhml/l3kmzYsKF99tln7oRsjqxaZdanj9mHHx64bvBgs+uuMxs40OzQQ/Nx7wGkilxlmqpWrWq7d+8OOr9J2wAg2ek98Pnnn08PmPytWrXKnTHX2s/A9T6abaeDQAImxJsVK1ZY3759g25TGWmksk3Lly+3Dh06ZGq9/8cff9jtt9/uZkDmyDffZAyYfF56yezPP/O4twBSVa6Cpuuvv9569erlhtr66P8qUdE2AEh2WsP0+eefh9w+ZswYV+oEJNJrOlRgpBMAvsZP+U3HD6GGIo8ePTpnwZpK8J54IvT2p54y2749F3sJINXlqjxvyJAhtnr1amvQoIErMdGbqRaMyrx582yw0uB+Z1wBILt0tllnvFWWo3/POOMM12RG5W3xRAvkw5UqqSGERjIAiUKv6XBKliwZkd8bbu2SSllVxZJtat8fJPubbv16sz17criHAJDLoGno0KE8dgAiEjB9/PHHdvHFF6eXtun9RmsblNWJp/JfBUW33nqr9ejRI+j2m2++2YoXLx71/QJyS63hmzRpYr+qaUKQgClSa5YbN24ccps6S+aoC6MarXTubPb008G3n3++WRK1+AcQ50GTDmgAIL8ps3TJJZdkWguktQ2PPvqoPfnkk25Yarxo3bq1nX/++a4Uz1+fPn3smGOOidl+AbmhdXZvvvmmmyu2XhkZvwzUqFGjIpbt1eiSU0891b799ttM21S5kqPfq+zuDTeYjRiROeOk5lXnnWdWoEA+7DWAVJOrNU0AEAnjx493HbyC0bDUvC5EVxmxFp0rOFPZT16pg+hLL71k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HJCKQdPYsWNd+1/NNtEZW5XpjR49OkM7cgCxUbduXdeme/Xq1Zm2qZlCqpbOqjOaSvX02ABaGxeuNPP888+3UqVK8UAhS8req5lIKD/99JML0BmkDaTgmiYtkL3jjjvcmbj169e7hhC6DkDsqZulStECGyZ0797dbrnlFtdQIZUsX77cxowZ47qq6THQGoNgASVSiwLoyy+/POg2rXXTSUA6KiI7ihUr5tbThaI1dwRMQIpmmjT7RQcevg8UBVDPP/98pPYNQC7W8KjduDo5aR1T5cqVXVcxHSimWsB09tlnuzUFPhp0qyzCCy+84Er09BiphEZlWIHd0tQsQMNTNUiVg57koiySWourMYpeIwqsCxYsaD179rR77rknx10akbp0Iqpfv3721ltvBd1+1113UQoMpGqmaevWre4gwkf/13UA4ofOljdp0sROO+00q1OnTsoFTFpv+f7772cImHx0gDxz5kx7+eWX7bjjjnMHyGrTrmG5KqPRLKgPP/zQLeLWek21oFYDCQVQSB6+NW066ac1uroomFYwBeSEMk1vvPFGhky+1s098cQTrrMjgBRuBPH6669ned1ll12Wt70CgFzSQNthw4aF3K6BqJqjogBJFES1adPGZsyY4Q5+tN1n5MiRNmrUKJsyZYo1b96c5yTJ6MRfdobhJvq6G2VV9a/WI2tUSKqV6kY6c3nhhRfaKaec4sY+KHtdu3Ztt4ZS88OC0fOh71Xljr5PJ7gAJGHQFKwWPPA6giYAsaLhkuEyQ9pWuHDhDNfpzLDWafoHTP4dsq655hobP348Q0+RUFSCeO+997ryMb2OVWp6880324033pgUjVEUCOo+6m9TJbmnn366a4ijwDDaa5u0fkmXrGg/VSaskzQ+Cq7eeecdd3/8q3kAJHB5nspesnMBgFhR5qBbt24ht3fq1Mm+++67TCU2P//8c8ifmTVrlmuAAyQKrctTsP+///3PBUyig3J1vX3sscfyNHh127Zt7hJL2v8vv/zSBUmaw6b7pYyNSms1iyse6YTN008/nSFg8p3okZUrV+bp9vfu3WuLFi1ymfa+ffvau+++a0uWLMnTbQLIZdAEAPFOWaRrr7026Jl0DZpUUPX3339nuF4lNYHZp0BqFgAkUpnqZ599FnSb1nKpRCyndFCvhirnnnuudenSxZWu/vPPP/bnn3+663/4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/v77b3cWVGdDRTOedL0/ZVoWLVqU4ToFWQrCNmzYkKErlG5n06ZNNmvWLHdRpsZn7dq1NmfOHJfpCqR98nWc0vfptgP3X/sRin7P3LlzM/y+YLetf/V9Wbn66qtdBkYXPUbr1693Qyl//fXXHP1u/33w7b/awvs/ntG673pulixZ4p7v7D5nAAAASAJpCWbWrFlpjz32WNp9992X9v7776ft3r07Rz+/adMmHfG6fwPt2LEj7a+//nL/BurevXta1apVg27z17lz57Sbbropw3W33npr2hlnnJH+9TvvvJNWrly5tBo1aqRVrFgxrWPHjmmrVq1Ke/7559PKly+fVqZMmbRjjz3WXaZPn562devWtK5du6YVLVo0rXr16mnFihVLu/feezP8jksvvTStZcuW7md0uyVLlnT/nzJlSvp1JUqUSDvzzDPT9u7dm/5ze/bsSbv55pvd99eqVcv926NHj7QtW7ZkuO1WrVqlnXDCCWlHHnlk2llnnZXl41yhQoW0Rx99NOz3ZOd3b9iwIa1Nmzbp971OnTruPvg/nsHue5MmTdKmTZuWL/f91FNPTTv55JPd7y9btmxao0aN0lasWOG2h3rOcvLaSgX6Ox07dmyO/16BSOD1iHjC6xHxJBVfj5vCxAb+EirTpBKoa6+91jZu3Oi+vu+++6xZs2a2efPmiLci//TTT61Hjx4u05TX+UnXXHONPfPMMy5bsnr1auvXr5/LUPXp08cuuOAC127Xl7U44YQTXPnfL7/8YgsWLHDfN3HiRBsyZIiNHTs2w21Pnz7dnnjiCXe7yrYsW7bM2rZta88995y7TtkxLaQeM2ZM+s8MGjTILbieP3++267MkDIr9957b4bb/u6771ymSFkbrTPIispp3n//ffvzzz9Dfk92fvf999/vMku6XvddQy8///zzTLflf9/1fcoItWnTJl/uu77nnnvucber79F6i8GDB7ttoZ4zAAAAJI+ECpquuuoq++GHH+zhhx+2gQMH2pQpU+yPP/4IehCdn1RappK4o446Ks+3pYYBGt5YsmTJ9OvatWvnSv9CeeWVV1zAqHVBoo5SF154ob388ssZvq9169YuSBKtJWrevLn7+pRTTnHXaWhv06ZN7bfffksP4J5//nnr3r27K2vTY6lgo3PnzpkCMu1fly5dsn0/X331VdceWIu59XvPO+88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SpWsffv2QR8JXa/tkTBv3jxr06aNC5hee+01O+ggXo4AAACIP7Qch2sr/uqrr2ZqBqGASddHqu34OeecY+vXr7e9e/faFVdckeH39uzZk2cGAAAAcYGgCY6Gy7733nvpc5pUkqcMUyTnNN1///22Y8eOLFuRAwAAALFE0IR0CpAiGSQF6tatG48+AAAA4h6LSAAAAAAgDIImAAAAAAiDoAkAAAAAwiBoAgAAAIAwCJoAAAAAIAyCJgAAAAAIg6AJAIA4sXXrVlu7dq3t3r071rsCAPBD0AQAQIytW7fOJk+ebF27drW2bdva7bffbvPmzbO9e/fGetcAAAy3BQAgtjZt2mTPPPOMDRo0KP2633//3V555RWbMmWKNWnSJKb7BwAg04QY01nUn3/+2T799FP766+/Yr07ABB1q1evzhAw+ezYscOuvfZaV64HAIitQjH+/Uhh48ePt5tuusnKli1r5cuXt2nTptlxxx1nY8eOtVKlSsV69wAgKqZOnRpy208//WQbNmywChUq8GwAQAwRNOGADRt0ylO1ImZly5pVrGhWrlxEH6FJkybZ4Ycf7v6vs6l16tSx559/3u68806eGQAwswIFCvA4AECMETTBs2yZ2VVXKf1z4BFp397s1VfNqlaNyKPUXrfvR2dSlXHasmULzwqAlNGyZcuQ20444QQrF+GTVwCArNE9D16GKTBgEn2t67U9QlSz/95779mIESNc16gyZcrYjTfeyLMCIGUcdthh9sADD2S6/uCDD7Zhw4a5k0kAgNhK2EzT5s2bXeMAlXYdeeSRsd6dxKaSvMCAyUfXa3uEznQqaNIaJj2fagihwKl06dIR+V0AEI/0nte3b19r3bq1PfbYY7ZixQpr06aNXXfddVajRo1Y7x4AIJGDposvvtg+++wzl5V4+umnY707iU1rmPKyPQ8OOeQQl2nyzSlp1qyZFSpUyLXfBYBUoffCU0891Y4//njbtWuXC6QKFy4c690CACRy0KQD6p07d1rDhg1jvSvJoUyZvG3PJypB6dixo33//fdR+X0AEG9KlizpLgCA+JJwa5pmzpzpyhdef/11Ogrll0qVvKYPweh6bY+AlStXZvg6LS3NPb9VI9R4AgAAAEj6TNPWrVute/furiX1EUccEevdSR5ar6QueaG650VoPdMll1xiNWvWdNPuNeT2ww8/tLlz59qLL74Ykd8HAAAAJFzQtHDhQluzZk3Y79Gw06JFi7r/X3/99daqVSs799xzs/07VBuui48aDsiePXvcxZ++VrZj//797pLIdD98/2brvlSubDZypBXQ86E1TGXKWNqhh3oBU4Qei3Hjxrn1TBpqqzkkXbp0sdGjR7sOeon++AfS/dFzoddYwYIFLdX4/tYC/+aAWOD1iHjC6xHxJBVfj3uyeV8LpPmOrmNg0KBB9vnnn4f9Hh1EV6lSxTV9UPOHMWPGWIkSJdy2Xr16uSzFzTffbC1atAj682rjOnDgwEzXjxw50rVz9acGBGr9qvKwIkWK5Om+Af52795ty5Yts1WrVrmsGgAAAGJv+/bt1rNnT9u0aVPYDs4xDZpyQsHT0KFDM1z3xx9/uDunluNTp04NegY/WKZJQdHatWszPTBqLqED2+rVq1uxYsUskelp1ZDYUqVKsfYrDui19c8//7jXXqK/tnJ7FmfChAnWrl07OoIh5ng9Ip7wekQ8ScXX4+bNm61ChQpZBk0Js6bpggsucBd/jRs3dnMtwrUcV2mfr7zPn14IgS+Gffv2uQDjoIMOcpdE5itv890fxJaeAz0XwV53qSTV7z/iC69HxBNej4gnqfR6LJzN+8nRNAAAAAAka9DUqFEjV0oHAAAAAJGSMOV5wbz55pux3gUAAAAASS6hM02RkiC9MZBAeE0BAAAkLoImP77ue2oPDeR3O0tJlUWVAAAAySShy/Pym+Y0aXaTBu7q4DaRu86pe56CP7W6TuT7kQwZJgVM//33n5UtWzYlB9sCAAAkOoImP2oJffjhh9vixYttyZIllugH6zt27LDixYszpykOKGDS4GQAAAAkHoKmAEWKFLFatWolfImehpN999131qpVK0rCYkxZSzJMAAAAiYugKQiVsxUrVswSmQ7S9+7d6+4H62gAAACA3GOxCwAAAACEQdAEAAAAAGEUStV5OZs3b7ZkpjVN6tqm+0l5HmKN1yPiCa9HxBNej4gnqfh63Pz/MUFWMzVTLmjasmWL+7dq1aqx3hUAAAAAcRIjlClTJuT2AmlZhVVJRvOLVqxYYaVKlUrqVtyKmhUYLlu2zEqXLh3r3UGK4/WIeMLrEfGE1yPiSSq+HtPS0lzAdMQRR4SdbZpymSY9GFWqVLFUoRd8qrzoEf94PSKe8HpEPOH1iHiSaq/HMmEyTD40ggAAAACAMAiaAAAAACAMgqYkVbRoUbv//vvdv0Cs8XpEPOH1iHjC6xHxhNdjaCnXCAIAAAAAcoJMEwAAAACEQdAEAAAAAGEQNAEAAABAGARNKWjWrFk2ZcqUWO8GUtyaNWts5syZtnHjxljvClLItm3b7JdffrFFixbFelcAW7VqlftM1kBRIF789NNPNmPGjFjvRtwhaEoxX3/9tTVr1sxOOeUU27t3b6x3BylIb8SnnnqqNWjQwK644go3gfvKK6+0PXv2xHrXkORGjhxphx9+uPXs2dMaN25sp512mm3atCnWu4UU9O2331qLFi3c6/Cyyy6zww47zG688Ubbv39/rHcNKe7dd991r83OnTvHelfiDkFTCvnvv//cQWrfvn1jvStIYfPnz7eHHnrIvR6Vafr999/t448/tkcffTTWu4YktmDBAndw+sQTT9jff/9tS5cutRUrVli/fv1ivWtI0dfjc889l55p0smkESNG2PPPPx/rXUMKW7hwod122212zTXXxHpX4hItx1OEOst36tTJWrdubZUrV7ZLLrnEndkvVKhQrHcNsB49eti6dets/PjxPBqICM2tGz58uC1fvtwKFCjgrnvhhResf//+tn79eitevDiPPGKqY8eOVqpUKfvggw94JhB1u3fvtpYtW9oNN9zgyucHDx5sa9eu5ZnwQ6YpRQwdOtR27NjhziAA8WTfvn0u43T00UfHeleQxPQaa9KkSXrAJM2bN7edO3fa3LlzY7pvwK5du+yPP/7gfRAxc+edd1qNGjXs0ksv5VkIgTRDAtq+fbv9+uuvYb9Hdfs1a9Z0///xxx9d0PTzzz/bQQcRJyP/F9brgDQcrVs66qijQmYA/v33X8qkEFHKJvneE33Kly+fvg2IpVtvvdWd2OzTpw9PBKJu3LhxNmrUKPvtt9949MMgaEpAq1evtjvuuCPs95xzzjmu7EQuvvhit/B5yZIl7jJv3jx3/dSpU91BRJUqVaKy30hOK1euzPL1eP755wcNilS///jjj9uHH35otWrViuBeItUVLlzYZZX86SBVihQpEqO9Aswefvhht57piy++cOXzQLSznMouab377Nmz3XX//POPaxamTsu1a9e2ihUr8qSwpik1tG3b1v1R+KhWVYGTalevu+46u+iii2K6f0hNL730kt18883u7NbZZ58d691B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",
      "text/plain": [
       "<Figure size 1000x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10, 7))\n",
    "import seaborn as sns\n",
    "sns.scatterplot(\n",
    "    x=\"Principal Component1\",\n",
    "    y=\"Principal Component2\",\n",
    "    hue=\"Customer_Segment\",\n",
    "    data=pca_df,\n",
    "    palette=([\"red\",\"black\"])\n",
    ")\n",
    "\n",
    "plt.title(\"PCA of Wine Dataset\")\n",
    "plt.xlabel(\"Principal Component 1\")\n",
    "plt.ylabel(\"Principal Component 2\")\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "4c616e33-1200-40ca-a898-325f0b11a4b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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WrYcAAAACJYRsMqU+URubq656d62bZtUqVKl0DrNwFilq4OLDSSl7HiRRoO1bW00qAUW9Fz1HEMbyJaEVtDr2jW/SIVeOtqg9ssvv5SlS5eK36Xs+nEhLes+ffp06devX5GGygDXI/yMfx/hJH68HnOjxAbB+LN6OmhgpGXFtUqeFn3QOUw6JM/mDFMyaAbniiuukObNm5vMlV5okydPTvdhAQAAAElD0JQuLm00+M9//tPMF9LeSTokT/snac8nAAAAwKsImpBQzyerYS0AAADgdZQcBwAAAIAoCJoAAAAAIAqCJgAAAACIgqAJAAAAAKIgaAIAAACAKAiaAAAAACAKSo77yNq1a+W7776TjIwMOfroo+XQQw8Vr/nrr7/kvffek1NOOaXY/lHr1q2Tr7/+WjIzM6Vt27ZRu0ADQNId2O/KpucA4AcETT5x0003yZNPPikdO3aU8uXLy9KlS6Vnz57y9NNPS5kyZcQrcnJyZODAgbJq1aqIQdPevXvl0ksvlY8++khatmxpfubHH3+Up556yvwsAKTc8ndEZowQyV37v2VVDhPpM1ak2al8IACQZgRNabL/wH5ZvGGxbMzbKIdmHipts9pK6ST9RVEzLw8//LB89dVX0qZNm/z9798vEyZMMAFEcND0559/yueffy6lSpUy24bLRi1fvlxWrlwpRxxxhDRs2FCmTp0qffv2lapVq8q2bdvk/fffl9NPP11Wr15tgpEGDRpIixYtzM9qZue3336TVq1ameWhou0/+LV///13+eGHH6R+/foFr63nMm3aNPP9u+++a362Vq1acsIJJxTah26nmagXX3zRZN3UnXfeaQKps88+Ww46iF8LACkOmCZdKCKBwstz1+UvP+dlAicASDPuDtNg9urZMmbhGFmft75gWc3MmjKy/UjpUa+H7ftbtmyZVK9evSBgUqVLl5YhQ4YU2k6DqOuuu85sp4HUF198IQ899JAMHTq0YJuRI0fKI488Il26dJE1a9aYoEUDGc1cadCkwYxmazRQ0SCnZs2aMnv2bPNzehx//PGHHHLIISbLo/s766yzYt6/9dqnnnqqySTVrVtX5s2bJzfccIPce++9JhiaNWuW2faDDz4wmabWrVsXCZp0OF7wftVRRx0le/bsMV8ETQBSOiRPM0yhAZOhyzJEZowUadKfoXoAkEYETWkImIbPGy6BkP8gN+RtMMsf6vaQ7YFThw4dZPPmzTJs2DC57LLLpGnTpgUZluDs0ZVXXilz5syR9u3bm2ULFiyQXr16yUknnST16tWTxYsXy/333y8ffvihdO/eXQ4cOFAk+LBosDJ+/Hjz/bhx4+SWW26RESNGyFtvvWWWjRo1Sv71r38V/Hws+7fo92+//bb5XjNLp512mjk3zSzpPnXZo48+agK6aBYtWmQyZpr50mGKGqBpQAUAKaNzmIKH5BUREMldk79dg658MACQJlTPS/GQPM0whQZMylo2duFYs52dNMDRoWgaTDRv3lyqVasmAwYMkPnz5xfK8tSuXdsEEG+88YZMmjTJFErQIOLTTz8120yZMkWOOeYY83pKh9DpXKlwrrjiioLvO3XqFHbZTz/9JIFAIOb9WzS4snTr1s0MNfz555/jfl90qKAOLXzttdfMfpo0aRL3awBAiWjRBzu3AwAkBZmmFNI5TMFD8sIFTjl5OWa7dtntbN23DsXTLw1KdM7QM888YwIOK2v066+/ys6dO2Xy5MmFfk6LRRx88MEFw+NCszeRsjkamFnKlSsXdtm+fftMwKPD4WLZv0WHGoa+9q5du+J+Ty6//HLzpTTLpBkrzTwddthhcb8WACREq+TZuR0AICkImlJIiz7YuV0iDj/8cPN15plnSuPGjeXVV181QZOW287OzjZZl0g0WNGS5cG2bt1qy3HFsv9kuuCCC0zW7MsvvzRzpgAgJbSsuFbJ06IPYec1ZeSv1+0AAGnj2uF599xzj8lA3HbbbeIWWiXPzu1ipcUXNKsTTDM8WjjByuL06dPHVNfTeT6h1ezy8vLM9507dzZD5TZt2lSw3ppbVFKx7D8WlSpVKjbztGHDBjMfK5juW2lxCQBIGa2aqmXFjcJzTQue9xlDEQgASDNXZpq0QMALL7xgqrDpkC630LLiWiVPiz6Em9eUIRlmvW5nJ61UpyW1dR6TVonTgEIzTLt37y6YH6RD07QynQ6Hu/76602xBS3O8M4775i5TzrnR7NTDzzwgCnMoPOTdKjfyy+/nH/sIYUl4hXL/mOh1fo08Pn3v/8t/fr1M/OkQqvnaUELLUKh+9T1WrpcezRdfPHFhSoMAkBKaB8mLSsetk/TGMqNA4ADuC7TtGXLFrnwwgtNYYPKlSuLm2gfJi0rbgVIwaznI9qPsL1f0/nnn28CJ+1ZpGW8V6xYIeedd57poaR9lsz+MzJMMQb90iF3n332mQk+Fi5caH7OKvygc6DOPfdckxHSIXVatEFZn4VmrnR9hQoVCg3r02Vly5YtWJaVlWWW6WvGuv9wr60/r8v09azXmTlzphnqp4Uv9HXCZbVeeeUVk33TgEx/RoOz559/3tb3HQDiCpxuXCYy5D2RM5/Lf7xxKQETADhERsAqX+YS2thUm5lqJkHLWmsxA6u0dSxyc3MLmrDqTX8wzcBo/x9tulq+fHlJZZ+m7MxsEzDZVW5ch5/pueo5WoGJXUFrcCEGfe+1R9L69etLnG1yu1RdP26kQ0GnT59usn/BzZQBrkf4Hf8+wkn8eD3mRokNXDs8T3vvrF27tkiFtWh0CJp+Bb8x1kWhX8H0ucaQGnCEznmx04l1T5QTap9gquRt2rlJalSoYYbkaYbJrv1asbB1PnYZPny4uaBatmxpikJof6MHH3zQ7Mdl8bft9H3W90CvI20ejP+xftdCf+eAdOB6hJNwPcJJ/Hg97o3xXNOaabr66qtl4sSJxfbS0b/cf/vtt3LiiSea4VZHHHGEWRdLpkmbqI4ePbrIct1v6DwZLX2tw7p0WFjwUDIUzqboELqlS5eajFPfvn0LmtH63Z49e0xZ9pycnCKFNwAAAOA8WnBs0KBBxWaa0ho06UHqjWY01vCyxx57TIYNGyYVK1YsWLd9+3YT6Ogcl82bN4f96364TJMGRVoBLtzwPKsXkduHV+nHqu+PzjXy+7C5VNHrR/tN6fXl9usnGX/FmTVrlin04Zd0P5yL6xFOwvUIJ/Hj9Zibmys1atRw9vA8zfTEWhVNm5AOHjy40LKuXbtKly5d5L777os4HEqbn1oNUIPphRB6MWgZbg0wNEizcx5QOlhD8qzzQfLp+6zvd7hrC/l4b+AkXI9wEq5HOImfrscyMZ6na+Y06XC50CFzGihpQGT1GgIAAAAAu5GCCMPvBQ2QGK4bAAAAb3JNpilSk1ud02QXa4ifzrMK7gUExDpHT/klnQ0AAOAXrg6a7G5uqwGYzrHauHGjufF181wgndOkwZ8WJ3Dzebglw6QB04YNG8xQUcqNAwAAeIurgya76ST+WrVqmQalq1evFrffyO/cudNkzKielxoaMGnJegAAAHgLQVMILTahfaCKK4XuhpKR8+fPl+OPP57hYimgmUkyTAAAAN5E0BSGDmdze58dvYHXBqt6HsyxAQAAABLHZBcAAAAAiIKgCQAAAACiOMivvXRyc3PFy3ROk1Z00/NkeB7SjesRTsL1CCfheoST+PF6zP07Jiiu36bvgqbt27ebx7p166b7UAAAAAA4JEaoWrVqxPUZgeLCKo/R/kVr1641PZ68XIpbo2YNDH///XepUqVKug8HPsf1CCfheoSTcD3CSfx4PQYCARMwHXbYYVF7m/ou06RvRp06dcQv9IL3y0UP5+N6hJNwPcJJuB7hJH67HqtGyTBZKAQBAAAAAFEQNAEAAABAFARNHlWuXDm58847zSOQblyPcBKuRzgJ1yOchOsxMt8VggAAAACAeJBpAgAAAIAoCJoAAAAAIAqCJgAAAACIgqDJh5YsWSIff/xxug8DPrdx40b5+uuv5c8//0z3ocBHduzYIV999ZX88ssv6T4UQHJycsz/ydpQFHCKRYsWyRdffJHuw3Acgiaf+fDDD6Vdu3bStWtX2bdvX7oPBz6k/xCfcMIJ0rx5cxk6dKjpwH3JJZfI3r17031o8LiJEydKrVq1ZNCgQdK6dWs58cQTZdu2bek+LPjQRx99JB07djTX4UUXXSTZ2dly/fXXy4EDB9J9aPC5V1991Vyb/fv3T/ehOA5Bk49s2LDB3KRed9116T4U+NhPP/0k//73v831qJmmb7/9Vt5++22577770n1o8LCVK1eam9MHH3xQfvjhB/ntt99k7dq1MmzYsHQfGnx6PT766KMFmSb9Y9ILL7wgjz32WLoPDT72888/yy233CJXXHFFug/FkSg57hNaWb5fv37SrVs3qV27tlxwwQXmL/sHHXRQug8NkIEDB8rmzZvlgw8+4N1AUmjfumeeeUbWrFkjGRkZZtnjjz8uN998s2zZskUqVKjAO4+06tu3r1SuXFkmTZrEJ4GU27Nnj3Tu3FmuvfZaM3x+zJgxsmnTJj6JIGSafGLcuHGyc+dO8xcEwEn2799vMk6NGzdO96HAw/Qaa9u2bUHApNq3by+7du2S77//Pq3HBuzevVuWLl3Kv4NIm9tuu00aNGggQ4YM4VOIgDSDC+Xl5cnixYujbqPj9hs1amS+X7hwoQmavvzySylVijgZ9k+s1xvSaHTeUsOGDSNmAP744w+GSSGpNJtk/ZtoOeSQQwrWAel00003mT9sXnPNNXwQSLn3339f3njjDfnmm29496MgaHKh9evXy8iRI6Nuc/rpp5thJ2rw4MFm4vPq1avN148//miWf/LJJ+Ymok6dOik5bnjTunXrir0ezzzzzLBBkY7ff+CBB+TNN9+UI444IolHCb8rU6aMySoF05tUVbZs2TQdFSByzz33mPlM06dPN8PngVRnOTW7pPPdv/vuO7Ps119/NcXCtNLykUceKVlZWXwozGnyh5NOOsn8Ulh0rKoGTjp29aqrrpLzzz8/rccHf3ryySflxhtvNH/dOvXUU9N9OPA4LQKhk++D2y3oHLrevXubG4R69eql9fjgTzpv5K677pL33nvPVHMEUu2vv/6SPn36FFqmcz91BEiHDh3k9ttvN/PtQCEIX5owYQKFIJBWTz/9tCmvqxOeTzvtND4NJN0rr7xiAie9GbD+aqpDoWbNmlWQfQdS6f7775dRo0bJu+++a/64CTiFTumgEERRDM8DkPKgXTOcOnxU55RYf/mvWLGitGnThk8DSXHuuefKQw89ZIL0W2+9VVasWGGC99dff513HCn3xBNPyIgRI2T06NFSrly5gn8Hq1atKi1btuQTARyIoMmH9K+sOjQvuIoUkCqrVq2S4447Tj799FPzZdFCES+//DIfBJJC2ytoc++xY8eaUuPVqlWTadOmmeF5QKrp/GL9f1iHiAa3WmjVqpUJqIB00rnu2uAWhdGnCQAAAACioP40AAAAAERB0AQAAAAAURA0AQAAAEAUBE0AAAAAEAVBEwAAAABEQdAEAAAAAFEQNAEAAABAFARNAOAzW7dulddee0327t3r6NdM5usCABAPmtsCgIu98847kpeXZ76vWLGiNGrUSJo1axb1Z7788ktp166dCUgOPvhgW44jGa8Z7+uuX79elixZYgIsfR+aNGkiGRkZth2Ll2zevFlmzZolZ511lhx00EFRt929e7d88MEHUrp0aenXr1/KjhEAnCT6v5QAAEe7+uqrpXr16iZQ+uuvv2T+/PnSsWNHE0yVL18+7M/o9ueee66ULVvWtuNIxmvG6s8//5SrrrpK3n77bWnfvr1UrVpVVq5cKWXKlJHHH39cOnfunPJjcrqffvpJBg4cKCeffLJUqlQp4nYjRoyQCRMmmMCqQoUKBE0AfIugCQBc7pxzzpE77rjDfK/BQqtWreSRRx6Riy++WObMmWPW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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,6))\n",
    "for seg in pca_df['Customer_Segment'].unique():\n",
    "    subset=pca_df[pca_df['Customer_Segment']==seg]\n",
    "\n",
    "    plt.scatter(\n",
    "        subset['Principal Component1'],\n",
    "        subset['Principal Component2'],\n",
    "        label=f'Segment {seg}'\n",
    "    )\n",
    "plt.title('pca on wine dataset')\n",
    "plt.xlabel('Principal Component 1')\n",
    "plt.ylabel('Principal Component 2')\n",
    "plt.legend(title='customer segment')\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "8a2c76a6-3c02-4f44-80dd-f51ffe049f0d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['Principal Component1', 'Principal Component2', 'Customer_Segment'], dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(pca_df.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bcd5cddb-a77e-4f9e-9f1f-974789edd159",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53453c8d-e1b7-4750-86df-c484e80fcd9c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "fa27af4a-7680-4ab1-ae20-28f58c1a11b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
    "from sklearn.metrics import mean_squared_error,r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "e2cfb261-47cd-4d21-a4f1-c690c19f9ec7",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('uber.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "53f1080b-fceb-447a-a2d7-fee6616339ab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Unnamed: 0                            key  fare_amount  \\\n",
      "0    24238194    2015-05-07 19:52:06.0000003          7.5   \n",
      "1    27835199    2009-07-17 20:04:56.0000002          7.7   \n",
      "2    44984355   2009-08-24 21:45:00.00000061         12.9   \n",
      "3    25894730    2009-06-26 08:22:21.0000001          5.3   \n",
      "4    17610152  2014-08-28 17:47:00.000000188         16.0   \n",
      "\n",
      "           pickup_datetime  pickup_longitude  pickup_latitude  \\\n",
      "0  2015-05-07 19:52:06 UTC        -73.999817        40.738354   \n",
      "1  2009-07-17 20:04:56 UTC        -73.994355        40.728225   \n",
      "2  2009-08-24 21:45:00 UTC        -74.005043        40.740770   \n",
      "3  2009-06-26 08:22:21 UTC        -73.976124        40.790844   \n",
      "4  2014-08-28 17:47:00 UTC        -73.925023        40.744085   \n",
      "\n",
      "   dropoff_longitude  dropoff_latitude  passenger_count  \n",
      "0         -73.999512         40.723217                1  \n",
      "1         -73.994710         40.750325                1  \n",
      "2         -73.962565         40.772647                1  \n",
      "3         -73.965316         40.803349                3  \n",
      "4         -73.973082         40.761247                5  \n"
     ]
    }
   ],
   "source": [
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "c31c2e9e-48d2-4396-8180-71d5725787ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 200000 entries, 0 to 199999\n",
      "Data columns (total 9 columns):\n",
      " #   Column             Non-Null Count   Dtype  \n",
      "---  ------             --------------   -----  \n",
      " 0   Unnamed: 0         200000 non-null  int64  \n",
      " 1   key                200000 non-null  object \n",
      " 2   fare_amount        200000 non-null  float64\n",
      " 3   pickup_datetime    200000 non-null  object \n",
      " 4   pickup_longitude   200000 non-null  float64\n",
      " 5   pickup_latitude    200000 non-null  float64\n",
      " 6   dropoff_longitude  199999 non-null  float64\n",
      " 7   dropoff_latitude   199999 non-null  float64\n",
      " 8   passenger_count    200000 non-null  int64  \n",
      "dtypes: float64(5), int64(2), object(2)\n",
      "memory usage: 13.7+ MB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "58bbed63-59e8-49d7-90ea-e6ac06a34f7e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "         Unnamed: 0    fare_amount  pickup_longitude  pickup_latitude  \\\n",
      "count  2.000000e+05  200000.000000     200000.000000    200000.000000   \n",
      "mean   2.771250e+07      11.359955        -72.527638        39.935885   \n",
      "std    1.601382e+07       9.901776         11.437787         7.720539   \n",
      "min    1.000000e+00     -52.000000      -1340.648410       -74.015515   \n",
      "25%    1.382535e+07       6.000000        -73.992065        40.734796   \n",
      "50%    2.774550e+07       8.500000        -73.981823        40.752592   \n",
      "75%    4.155530e+07      12.500000        -73.967154        40.767158   \n",
      "max    5.542357e+07     499.000000         57.418457      1644.421482   \n",
      "\n",
      "       dropoff_longitude  dropoff_latitude  passenger_count  \n",
      "count      199999.000000     199999.000000    200000.000000  \n",
      "mean          -72.525292         39.923890         1.684535  \n",
      "std            13.117408          6.794829         1.385997  \n",
      "min         -3356.666300       -881.985513         0.000000  \n",
      "25%           -73.991407         40.733823         1.000000  \n",
      "50%           -73.980093         40.753042         1.000000  \n",
      "75%           -73.963658         40.768001         2.000000  \n",
      "max          1153.572603        872.697628       208.000000  \n"
     ]
    }
   ],
   "source": [
    "print(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "3e590998-1880-41ca-84e0-7f8bff9e3ebe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(200000, 9)\n"
     ]
    }
   ],
   "source": [
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "8a786cf7-60ea-42a9-bc8e-7631428e97fc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Unnamed: 0', 'key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']\n"
     ]
    }
   ],
   "source": [
    "print(df.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "77a1139c-463b-4981-abdc-1441548cf577",
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_columns=['fare_amount', \n",
    "'pickup_longitude', 'pickup_latitude', \n",
    "'dropoff_longitude', 'dropoff_latitude', 'passenger_count']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "51bbd143-cd93-48e4-9070-07d251fff0b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "for col in numeric_columns:\n",
    "    df[col]=pd.to_numeric(df[col], errors='coerce')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "4947abfd-e0a2-46ec-8ee0-973bf673f3b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'], errors='coerce')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "6fee7d70-48de-4940-b383-28b7f1dd0771",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['pickup_hour']=df['pickup_datetime'].dt.hour"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "4e13d09c-2180-4056-af37-3b93e3614c3a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=df.drop(columns=['Unnamed: 0', 'key', 'pickup_datetime'], errors='ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "c15886d9-de62-4eda-934e-e64550af3b7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=df[\n",
    "(df['fare_amount']>0) &\n",
    "(df['passenger_count']>0) &\n",
    "(df['pickup_longitude'].between(-180,180))&\n",
    "(df['dropoff_longitude'].between(-180,180))&\n",
    "(df['pickup_latitude'].between(-90,90))&\n",
    "(df['dropoff_latitude'].between(-90,90))\n",
    "]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "60273644-be3c-4e3a-8623-b3a56487e2ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=df.fillna(df.mean(numeric_only=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "15025731-4c02-4cfb-9207-b7cdf4e7053d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=df.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "fe3d3d91-8f74-45c8-a1c1-c1f1af815b60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fare_amount          0\n",
      "pickup_longitude     0\n",
      "pickup_latitude      0\n",
      "dropoff_longitude    0\n",
      "dropoff_latitude     0\n",
      "passenger_count      0\n",
      "pickup_hour          0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "e190b5d6-daf5-477c-878a-2ae4c221132b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAoQAAAGICAYAAADYnfGpAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvctoD+AAAAAlwSFlzAAAPYQAAD2EBqD+naQAALVtJREFUeJzt3QeUFFWixvFLzkFAcgYlKgqCIGYWTKyiiGDARQkqKK7hKYoLgiK7ophzQEmigqggBhRckKSSQQQVECQjOYepd777XrXVPT0z3TDN9Mz9/84ZZrr6VtWtqunpjxuqc3me5xkAAAA4K3dWVwAAAABZi0AIAADgOAIhAACA4wiEAAAAjiMQAgAAOI5ACAAA4DgCIQAAgOMIhAAAAI4jEAI5wN13320uueSSTN/uwoULTceOHc0ZZ5xh6tata7Zu3Zrp+4AxX331lT2/P//8s7PH69o5AJINgRCIwX333WffrKJ9vfHGG1l+DtetW2d+++23TN3mrl27zN/+9jeTO3du8+6775qPP/7YnHTSSZm6D/x1rpcvX24OHDiQY07JhAkT7Ovjl19+iel4c+I5ALKTvFldASA7UODSm9WSJUtMnjx5wp4rW7asyYlmz55tWwTvuusu06hRo6yuTo6m1t1ly5aZGjVqmJxi586d9jVz8OBBZ88BkJ0QCIE41KlTx+TN68bLZuPGjfZ78eLFs7oqOV6xYsVsa5rLOAdA1qLLGMgEq1atCutG1pi7a6+91kycODGs3DfffGOf19i8Tz75xFx66aWmfv36tgVS9P2BBx4w55xzjm2V69Chg/nuu+9irse+ffvMgw8+aJo2bWrOO+888/zzz5ujR4+mKpfRfq688kq7Hf9n1fnOO+8MPb948WLTtWtX06RJE3PWWWeZHj162NagaOMa1QXYr18/07x5c3PjjTeGnp8yZYq5/vrrzZlnnmnOPvtsc//995vNmzene3yHDh2ydXnxxRfNTz/9ZLen+r///vv2ebVG6Zhbt25tTj/9dPv9zTffNJ7nhbYxduxYu401a9bYrvALLrjAtGjRwjz77LOhMu+884656KKL7PH179/fHDlyJFVdtm3bZh555BHTsmVLW4drrrnGfPbZZ6HnZ8yYYfcTXOZT976e037SGj8XrKeOT/XRuerdu7fZvn17qm2uXbvWXgeVadWqlV1fv2fahn7vMpLR8YjOu7an6xA0b948u/zbb7+1j1999VX7+xX8/dGXjjMtaY0hjPeajho1yh6/Xld6PejaqT6XXXaZPTdt27Y1L7zwQqpjAFznRlMHkGCVKlWyY+yCb66ffvqpfTPUG5SCj+zevdsGp7feesvs3bvX9O3b13z//ff2TWvRokXm4osvtq2QDz/8sDn55JPN+PHj7bKRI0ea6667LsN6dOnSxZZ/6aWXbMD7n//5HxsKtD9fLPt57rnnzJgxY+zz+vmUU06xLTiicHHFFVfYN90nn3zSpKSkmP/85z82POlNXSEzOK5RdVI4HTp0qJk6dap9Tutp2wqNCpp64x44cKBp1qyZPR9pdcNrXzp/P/74oz3f//znP02uXLnM/v377TZUp99//908/vjjNjzo2BVM5syZExrruWPHDruNZ555xhQoUMDWRYHG35YCyJYtW8zgwYPtuVIAkwEDBoTqoecVYv1j0fVXKFHY+Pe//23DtJ7XuLhXXnnFnq+gt99+2/z666+mTZs2aY6f8+v58ssv2/DyxBNP2NDXq1cvOy7v888/D5XdsGGD3V/p0qVtuVKlSpnhw4fbgKZt6PcuPbEcj2gIgban6xCkc6/le/bssY/1HwztU+fe//0RbTct0c5BvNf06aefto/1u/TFF1/Yeuo/GjoXut4Kups2bbLP6bU3ZMiQdM8L4BQPQIY6duyo5givTp06YV9NmzZNd72bbrrJa9iwYejx+PHj7XYuvfTSsHIpKSlekyZNvNq1a3v79+8Pe6579+5e2bJlvYMHD6a5n/bt23u5cuXyRo4cGbb8qaeesvubOXNmaFms+xk2bJhdd/HixWH1rFWrlle3bl3vyJEjoeWHDh3yatSo4TVo0CBVnYYPHx62/tKlS+3yPn36hO1/9+7ddv+9evVK8zhVZ9WpRIkS3o4dO8K2+/DDD3t58uTxFixYELbOuHHj7DqzZs2yj9944w37uEePHmHlOnfu7JUpU8a7/fbbw5bfcsstXsmSJe0+fHfccYfd14oVK8LKdu3a1S5fvXq1faxj1ON169aFyui8VapUyWvbtm1o2YcffmjrNH/+/NAyv57dunUL28cLL7xgly9btiy0TMdSoEABb/369WFlr7jiCltWv3fpifV4+vfvb7cX+bszffp0u3zChAmhZSNGjEj1+5Pe8UZbFu811estSNdM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kXDLli1T/R08cOCA+emnn+yHQpxyyilR14+lDDLXvn37zLx581ItP+2001LdA1Blly1bZkqWLGlq1aqV5vYyKpPpvCQ3b948r1q1al6lSpW88uXLe7Vq1fIWL16c1dXC/3v++ee9evXqeSVLlvTq1KkT9bzMnDnTXr8qVap4ZcuW9erWrestX7487jLIfD/99JN3xRVXeCeddJJ3xhlneEWKFPHatWvn7dy5M6zclClTvHLlynnVq1f3Spcu7Z1++une6tWr4y6DzLdt2zavW7duXqlSpbzGjRvb77qWurZBkyZNstelZs2a9no3adLEW79+fdxlkFhTp0718uXLp8me3v79+8Oe++STT+x1qV27tv2b27x5c2/Tpk1xl0HmW7x4sb1mzZo181q2bBn6UoYJGjNmjFe8eHHvlFNO8YoVK+ZdcMEF9jUcb5lESOpAeOjQIfuH6R//+Id9nJKS4nXo0MGGhaNHj2Z19eB53j333OMtXbrU69u3b9RAuG/fPq9ixYpez5497eMjR454l19+uX3jiqcMEkNvHhMnTgw93rhxo33Nde/ePbRM4VAh4YEHHgi9Li+66CLvvPPOi6sMEkPB77333gv9TTxw4IDXunVr+2bk27Jli31jGThwoH2soKGgcNlll8VVBomla6AGkHvvvTdVIFQwL1y4sPfkk0/ax3v37rV/I6+++uq4yiCxgXDDhg1pllm5cqWXP39+7+WXXw793axfv77XuXPnuMo4GQi/+uore4J//fXX0LIFCxbYZdOnT8/SuiFcWoHwo48+8nLlyhXWyvDdd9/Zazh//vyYy+DEuf/++22rr2/48OG2xWL79u2hZV988YW9Pr/88kvMZXDi9OvXz7bU+vTmoqCggOAbO3Zs2OsuljJILLXWP/744zbgRwbCoUOH2lajgwcPhpaNHDnSy5Mnj7d169aYyyCxgXDGjBn2fWv37t2pyug/W+oBCzZovfrqq16BAgW8PXv2xFwmUXIn+7g09b0H+88bNWpk8ufPb59D8tN10gdqV6hQIbSsWbNmoediLYMT58cffzS1a9cOPdY1qFmzph3Lktb1iaUMEmvRokXmv//9r3n11VfNa6+9ZgYMGBB2DTWms3DhwmHXR40CCxYsiLkMEueZZ54xO3fuNH369In6vK6PxqPp/S94fTTGXtc+1jJIrOuuu87ccMMN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",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,4))\n",
    "sns.boxplot(x=df['fare_amount'])\n",
    "plt.title('Fare before removing outliers')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "143660ae-9d07-4382-ad8f-0d87f2e6297b",
   "metadata": {},
   "outputs": [],
   "source": [
    "Q1=df['fare_amount'].quantile(0.25)\n",
    "Q3=df['fare_amount'].quantile(0.75)\n",
    "\n",
    "IQR=Q3-Q1\n",
    "\n",
    "Lower=Q1-1.5*IQR\n",
    "Upper=Q3+1.5*IQR\n",
    "\n",
    "df=df[(df['fare_amount']>=Lower) & (df['fare_amount']<=Upper)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "b23ba70a-38a0-4826-9f54-5273a30acc0b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAoQAAAGICAYAAADYnfGpAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvctoD+AAAAAlwSFlzAAAPYQAAD2EBqD+naQAAI0VJREFUeJzt3QeUXGX9P+A3jSQECE1CCWASJCAIUkWCIIaSI0iTLkF6B7GD+ANEEGlSFJFeQpPeQakCEQQEFDg0JYQioZtQ0kjmf77v/8y4u9nNbpLt7/Ocs2d37tyZ+957Z+d+5m3To1KpVBIAAMXq2dEFAACgYwmEAACFEwgBAAonEAIAFE4gBAAonEAIAFA4gRAAoHACIQBA4QRCoObvf/972mGHHdKXv/zltNJKK6WPPvrI0WklDzzwQD6m//jHP4rd39KOAXQlAiHMo0MOOSRf5Br7ueyyy7rM8f3www/TJptskuaff/506aWXpptuuikNGDAgrb766unEE0/s6OJ1eR9//HF68cUX0+TJk1N38ec//zm/zp977rkW7W93PAbQXfTu6AJAV/fGG2+kf/3rX+nZZ5+d5b5BgwalruKhhx5K//3vf9P3vve9HAKr4gL+9ttvd2jZuoONN944Pf/88+nzn/986i4mTZo0RwGvOx4D6C4EQmglUVPSlU2YMCH/XmihhTq6KN1S1LZ29dfIvHIMoPPSZAxt7IUXXqg1Ia+88sppjTXWSDvuuGO666676q13xx135HVi/euuuy5tttlm6Ytf/GJ677338v2vv/56+uEPf5i++tWv5j5+O+20U3r00Ueb3f7JJ59c2348Xzz+8MMPT6+99lptnc033zwdffTRtb9j3YMOOij/njZtWm76rj7HvvvuW+/5H3zwwfSd73wn79e6666bvv/979fCZVU87je/+U2uTdp9991zDeTsmtNnt36U5+yzz87lXG211XIz9x/+8Ic0c+bM2uNvvfXW/Bwvv/xyuvLKK3PN1HrrrZdOOeWUVKlU8jpXXHFF+sY3vpHWXHPNdNRRR+XnbShqTI855pi0wQYb5DJss8026eabb67d/8QTT+Tt3HDDDY3WHMf5Pu+885rsP1e3nPEcI0eOzMcxjn31vNf1n//8J98X68Q+XXXVVbXXV7x+mtPc/oTzzz+/0f6j0Swcy6OZOFx88cW5Njlsv/32tddH7FNTmupDOKfn9Jprrkmbbrppfj3HPs2YMSNdcMEF6Zvf/GY+NvH79NNPT1OnTm32mAD/nxpCaGNDhgzJ/fGq3n///Xzxj4vW9ddfn7bddtt6zW/nnHNOmjJlSg5oEbY+++yzPNgjLpJxEf+///u/tNhii6Vrr702bbjhhvn31ltv3eT299xzz7TVVlvlv+O5xo0bl0466aS0/vrrp2eeeSYtssgieZtjxoxJxx57bP57ueWWy7U5hx12WN7mt771rXTkkUfm51hggQVqz33mmWfmkBoh5be//W2+AJ9wwglp7bXXTo899lhaeuml83qxX0899VS67bbbcmDcddddc5/FpjS1fhyXCMpx//HHH58v/tFU/+Mf/ziNHTs270OYOHFiXidCRq9evXIfyFjv0EMPzSGjX79++ThEWaMJM/qBxrGJ41L1wQcf5PAczaERJJdffvl044035vMVoSp+IkxGmX7/+9+n7bbbrt4+RGCKMkR5m+o/Vy1nhJlPP/00/fKXv0xvvfVWOvjgg3O57r///tq67777bg61cV5+/etfp8997nM51P71r3/NzxGvn9lpyf5UX5/xfBGy6orH1d1OvOYiNEYoPO2009Iqq6ySl1fPeWMaOwZzek5/97vf5QAZ5Y3jE+ctAn28biMExjmJYxXB9Sc/+Ul+jQItUAHmydZbbx1VTpXhw4fX+9lggw1m+7jtt9++svbaa9duX3XVVfl5ttlmm3rrzZgxo7LKKqtUvvjFL1amTp1a777Ro0dXlllmmcpnn302R2WeNGlSpU+fPpUzzjijtuycc87J23/55Zfrrdu3b9/K9773vVmeI9br1atX5fDDD6+3/NNPP81l2nvvvWvL4nkXWGCByrvvvltbNnPmzCbL19T6xx13XKVnz56Vxx57rN76t99+e37M/fffn2+PGTMm3959993rrbfvvvtWFl100cpee+1Vb/mBBx6Ytzd9+vTasu9///uVHj16VJ555pl66x588MG5DC+++GK+feyxx+b1xo0bV6+sQ4cOrWyyySa1Zbfeemsu0yOPPFJbVi3nbrvtVm8b559/fl7+1FNP1ZYddthh+ZyNHz++3rrf/va387rx+pmdlu7PiSeemJ/vww8/rLfe448/npdfe+21tWXxdyyL+xpqbH8bWzan5zT+b+qKYz1s2LDKnnvuOUsZpk2b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z02Cm2yySf7UGJ3KH3zwwWanp6DreeONN3IfophOJjz55JP5Q2CEwJhlIPz6179OO++8c74dHwjj9uqrr5523HHHDi49bXnuo2l41KhRubvIiiuumAcTRj/izTffPP/QNU2aNCmNHDkydwOJa3uMIK9aZZVV8owiwf99++r0E1PHhJUxcWm8YKKzabwxqBXo/mbOnJkuv/zydPvtt+fagBhZGFNUVEeg0X3ENFIx4XhDcaGID4NVMcjoggsuyO8J0d/opz/9af7GCrr3uY9vqTj77LPzfHVLLLFEfv/fbbfdmp2qhs4r5h8cPXp0o/fFHKPrr79+7bb/+/bT6QMhAABty0csAIDCCYQAAIUTCAEACicQAgAUTiAEACicQAgAUDiBEACgcJ36m0qA7iEmFY5vo1h88cXTRhtt1NHFAaABNYRAmzrxxBPTxhtvnK666qr0yCOPONqtFLDvvfdexxJoNb6pBGhTQ4YMSUcffXTac889HelWcuyxx6Z77rknPfzww44p0Co0GQNt4qOPPsrfRT1hwoT03HPPpauvvjp95StfScsss0y64YYb8jp9+vTJ30+9xhpr1Ptu2vfffz/dfffdafvtt0//+Mc/8neffu1rX0uDBg3K90fz8/PPP5+WWmqp/Nj55puvRWWK78V97LHH8t8LLLBAWmWVVXJgreuf//xn+uCDD9J6662Xnn766fTuu++mESNG5O9Snzx5cho7dmz+ru2vfvWracEFF6z32Pgm0Pje9ddffz0tv/zyaa211qp3/+OPP54+++yz/Niq2JdXX301bb755vW2H+vE9uPvtddeO33uc5/L97/88su5hvC9997LxzTEurE9gLklEAJtFghvuummHID+9re/pTfeeCMHuOhHGMvDtGnT0t///ve05JJLprvuuistssgitdCzyy67pMsvvzyHq5VWWin/xP177LFHbi5dd9110/jx49P06dPTLbfckr7whS80W6ZYv7rtSZMmpYceeijtv//+6dRTT62tc+WVV6brr78+h74VVlghvfXWW+m1115LZ511VjruuOPysihThMMIl7E/1f3dcsstc8Bbc801czBcffXV080335zmn3/+vM65556bPv7443qB8NZbb837WQ2Esf0bb7wx9evXLwfgiRMnphdeeCEfn3hchOO4HaG5ui+DBw8WCIF5UwFoQwMGDKjceOONTd4/bdq0yoYbblg54ogjasseeeSRSrw9HXLIIfXWPfrooyvrrrtu5ZNPPqktO+CAAyojR46cq7L9+9//riywwAKVRx99tLbspz/9aaVnz561ZTNmzKisueaalT59+lSefPLJvGz69OmV4cOHV0444YTa46L8w4YNq7z33nv59oQJEyqDBw+u/OIXv6its/fee1d22mmnemU45ZRTKquvvvos2x87dmxt2c4771z55je/Wbt9zDHHVEaMGDFX+wzQGDWEQIeIpuCosfv0009zM3K1KbeuQw89tN7tiy++ONek3XHHHbl5Nn6WWGKJdNFFF+WayN69m39Li5q9qJWMpux4TNROxrajObvqy1/+cu12NGVHbWTU2EXzdIjtrLPOOumll16qPSaabw866KC02GKL5dtRu7fPPvvk5dGHck5EzeL6669fu/31r389nXLKKXP0HABzQiAE2tWHH36YNttss9yEHAFroYUWysEqmn4biibmqhkzZuSm2mgujebZurbddtsc9Br26Wvo/vvvTzvuuGMOgUOHDk39+/fPTbLvvPNOvfWqTddVffv2bXRZNP+GCKZRtnjOuoYNG5ZD75yK/ooNtzVlypQ5fh6AlhIIgXb129/+Noe/6JcXg0rCz3/+81p/uLp69OhR+7tXr15pwIABaYcddkiHHXbYXG37hz/8Ydpvv/3SCSecUFu26qqr5kA3L6KcEeJiAEhdcbvax7Ba2xh9E+sS9IDOwDyEQLuKptqoSauGwaj5aywMNmbUqFHpggsuyE29db355pst3vbw4cNrt2P0c9Q4toYNNtigNnq66rrrrsvLq6Jp/F//+le9dR544IE53laMkBYkgdakhhBoV1tvvXUejRu1gssuu2y64oorcqCLsNSc0047LU8/E6NtR48enWvmHnzwwfz7mmuuafbx22yzTd5uNDlH38Uzzjgj1zq2hl/96le53+FOO+2URo4cme688848Pcwll1xSW2fnnXfOtZMHHHBA7oMY68Q0M0svvfQcbSumoTnqqKNy+aP527QzwLwSCIE2FXMJxrQoVdVBIVGbFjV2MWF1NKtGsKuK2xGsqrWIVTHX3jPPPJMuvfTSPEdf9D/cddddc9BriTPPPDM3Ecd8gPHYP/7xj+m+++7LU9rUHdDRsC9iTCPTMLTFQJO6tXTxHFGmqMGMCaNjO7G95ZZbrrZO1E4++uij6bLLLktPPfVU3seYRqfuBNONbT/mSowgXXeQyZgxY3LZY0of084A88o3lQAAFE4NIdBtRK1jTHbdmJjGpW5tHQD/IxAC3UZ8Vd4nn3zS6H3R3CwQAjROkzEAQOFMOwMAUDiBEACgcAIhAEDhBEIAgMIJhAAAhRMIAQAKJxACABROIAQAKJxACACQyvb/AHYRkNdC+GSMAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,4))\n",
    "sns.boxplot(x=df['fare_amount'])\n",
    "plt.title('Fare after removing outliers')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "1b8ba8e3-a317-4ce9-a49a-5d3d112b7c94",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   fare_amount  pickup_longitude  pickup_latitude  \\\n",
      "fare_amount           1.000000          0.000132        -0.000589   \n",
      "pickup_longitude      0.000132          1.000000        -0.980175   \n",
      "pickup_latitude      -0.000589         -0.980175         1.000000   \n",
      "dropoff_longitude    -0.000962          0.955110        -0.942688   \n",
      "dropoff_latitude      0.000544         -0.943006         0.962615   \n",
      "passenger_count       0.012002         -0.000809        -0.000638   \n",
      "pickup_hour          -0.013295          0.002092        -0.002281   \n",
      "\n",
      "                   dropoff_longitude  dropoff_latitude  passenger_count  \\\n",
      "fare_amount                -0.000962          0.000544         0.012002   \n",
      "pickup_longitude            0.955110         -0.943006        -0.000809   \n",
      "pickup_latitude            -0.942688          0.962615        -0.000638   \n",
      "dropoff_longitude           1.000000         -0.980726        -0.000107   \n",
      "dropoff_latitude           -0.980726          1.000000        -0.001751   \n",
      "passenger_count            -0.000107         -0.001751         1.000000   \n",
      "pickup_hour                 0.001653         -0.002119         0.013291   \n",
      "\n",
      "                   pickup_hour  \n",
      "fare_amount          -0.013295  \n",
      "pickup_longitude      0.002092  \n",
      "pickup_latitude      -0.002281  \n",
      "dropoff_longitude     0.001653  \n",
      "dropoff_latitude     -0.002119  \n",
      "passenger_count       0.013291  \n",
      "pickup_hour           1.000000  \n"
     ]
    }
   ],
   "source": [
    "correlation=df.corr(numeric_only=True)\n",
    "print(correlation)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "13936798-ffd7-437b-bb67-fc537b82b925",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAogAAAIiCAYAAABLzjkyAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvctoD+AAAAAlwSFlzAAAPYQAAD2EBqD+naQAA9gFJREFUeJzsnQV0FFcbht8QkpAgEdzd3V2LtcUdChQo7qVIKVDcCsXdneLw4w7Frbg7BAkQnChJ/vPeMJvdZJNsshsIy/ecMyfZ2XvvyO7OvPezsQkKCgqCIAiCIAiCIHwijvaPIAiCIAiCIIhAFARBEARBEMIgFkRBEARBEATBABGIgiAIgiAIggEiEAVBEARBEAQDRCAKgiAIgiAIBohAFARBEARBEAwQgSgIgiAIgiAYIAJREISvktu3b+Off/6Bt7f3VzGuKVy9elVt29/f/7NvWxAEQR8RiIIgGPDu3TscPHgQGzZswKFDh+Du7h4rz9Du3bvRtGlTeHp6RrnvjRs3lBDz8fGx6LjmsmnTJrXtDx8+fPZtR3ROBEH49hCBKAiC4vnz52jZsiWSJEmCHj16YMmSJRg0aBAyZ86M4sWLY9euXVZzprZt26aE2OvXr8O8lyVLFjRu3BhOTk74lojonAiC8O0R90vvgCAIX56nT5+iRIkSsLOzw4kTJ1CgQAED4dizZ0+sX78eVatWhbVTuXJltQiCIHzLiEAUBAFdu3bFkydPcOnSJWTNmtXgjCRNmhTLly/HyZMnw5ypa9euqSVevHgoUqSIsj7qs3PnTvVe+fLllQg9ffq0Gi9fvnzKnVq4cGG1vcuXL6txaKlMkyaN6ssYwFOnTuHFixdInTq1Gt/W1jbCT+vRo0fKLU5sbGyUFTB37tzIlCmTrs3Zs2fVQrgPzs7O6v9KlSohWbJkKgaR261duzYcHR0Nxqfbme/RDZs9e3bkzJkzTAzh+fPnUb9+fQQFBal98fLyQsmSJcOcm8jw8/MzqX9k58kS50T/uAICAnTjlStXDg4ODup/npMjR47A19cXZcuWRcKECaP82YQ+h9q2eC5KlSoFV1fXKJ1DQRCijwhEQfjGoXCjdZA35NDiUJ9ixYrp/n/27BmaNWuGY8eOKfHy6tUrJS779u2L4cOH69r169cPKVKkUAJwxowZSmxkzJgRQ4cOVe7MCRMmKOF57949JYTo0m7QoAEWLlyI3377TYmibNmyKcFAwcH9zJEjR7j7SJG7ceNG9T8FmoeHB44ePaosn+xrb2+PixcvqoVs375dCVhC0cr9Ywxip06d8PDhQ51YJUOGDMHo0aOVqEmcOLE6dp6TlStXInny5Dpx1b9/f3VMnTt3VvtPwUlxtGrVKtSoUcOkz+TmzZsm9TflPFninGjHRQHIyQQ/0wsXLiiRx3hVLTwhVapU6ntAQcf1FNFR+Wz0zyGFIc8BRS/PwePHjzFr1iw0b97cpHMoCIKZBAmC8E2zcePGIF4Kxo8fb3KfihUrBiVJkiToypUrunWzZ89W40ybNk23Ln/+/EHp06cP6tmzZ1BgYKBad+vWraCHDx+qtpkzZw7aunWrWh8QEBB09+7doE2bNqn3hg4dquvj7e0dVK1aNdXe19dXrZs5c6Zqx7Ei4vr160Gurq5BQ4YM0a2bOHGi6vvkyZMw7Y2NO2vWrDDHxnGTJ08eVKZMGd260aNHq3b169cPevbsmVrn5+cXVKFChaAMGTIE+fv7R7ivUelv6nmyxDnR3y9PT0+17tWrV0GpU6cOaty4cVDdunWDXrx4oda/fv06KG3atEH16tWL8FjD2w9tWzVq1Ajy8PBQ63h8HTp0CIobN27QuXPnIh1XEATzkSQVQfjGoWuS0CpmCrRS7d+/X8Ul6rtY27dvr1zGf//9d5jxaTGkW5Ew6UWDVq8ffvhB/R8nThxkyJABw4YNU9YvWhO1PrRo0XpHS9LWrVsj3Ue6qzdv3qysbv/99x/Sp0+Pffv2IbpMmjQJefPmRZcuXQz2vXfv3jh8+HAY9zutXHSlE8Z1tm3bVllJuV+mYEr/qJ4nS5yTFi1awM3NTf3v4uKCOnXqqPF++ukn3feH7ul69eqp8AJjmLofPAe0XhIe39ixY5WVcfr06VHaZ0EQooe4mAXhGyd+/Pjqr6l1/86dO6f+Ml4wNFxHV/Lbt2+RKFEinZDS/g8N4+X0oWuSooGxbevWrVOuSMK/fI/QhVm3bl2j4925c0e5qOmi1WIiGY9Hkfr+/XtEB8bWUdT88ssvRo9XOyf6LviCBQsatEubNq36S7d1njx5It1mZP2jcp4seU70k5cI3b/hrWepHoYeaHGDUd2P0N8NCk+GQGixkoIgxCwiEAXhG0cTLIwTNAVNgIRO4CBaaRitDdEsYcYI/R4THChyGBe5du3aMO1ZfoZlaMKDljaKkrt37xokdTDZ4sGDB4gOWtFqU4+XaEkeGlp8HY/PFCLrH5XzZMlzEnq/aN2MaL3+8UZ1P4ydb65jrKogCDGPCERB+MahQORCoTFu3DijN2by5s0bJQSYZEJoCSpdunSYYsu0FmpuSKK5P40R+j0mWDDhg5YiFm2OKkx6oKVPX4AwE5YWQP26hhHtU2i4T3Sf8nhDw+Ml2jn5XETlPMXEOYkOpu6Hxq1bt1TSi0ZgYKCyQjIpShCEmEdiEAVBwNSpU1UJF8YRfvz4McwZWb16NQYPHqz+ZwmTlClTKley/iPheKNnsWVarxhPGF3atWun4tdYuiU0L1++VO7r8KCgoIVKn3nz5oUp/qyJFIpeU+Ax7dmzx8DKyvM0ZcoUFSdHK9jnxtTzFFPnJKqYuh8aCxYsMHhNIczs+SZNmsTI/gmCYIhYEAVBQIUKFVR5kdatWytrIkvYMGGELsH//e9/qmTJqFGj1Jli3btly5apOoGsb8gEBbabPHmyKgHDZAJz+PPPP3H9+nUVX0eLU/78+VU8G8uq7N27FwcOHAg3ppFldliihv1YN491F+m+rF69uuqvwbFpKWV7xsXRJarV/DPGyJEjVVkbHi+fMkOL4ooVK1RpGD6SMDyra0xi6nmKqXMSVUzdDw3GNTIJ5vvvv1dJN/x+NWzYUASiIHwmxIIoCIKC2cS08NBSSAsULWb3799XiQ7MoOUNXoPCgRbDmjVrqgxeJk+MHz9e/a9fzJg3f4rP0NClSKscE1hCQ2FCiyUf7UdXKuvhcXyOw21q7lxjj8Tr2LGjEkfsR0HHrGrW3qP40a8hyKQPCihmVO/YsUO10bK5jY3LjN3jx49j4sSJ6tnUfNoMhQv3p0qVKrp2uXLlUn21mEF96xzX69dVNEZU+pt6nixxTsLbL2ZRc71WLFuD9Q+5Xl84m7ofGq1atVJinFnz/D4uXrxYZT4LgvB5sGGtm8+0LUEQBEGIkDFjxqhC2bRKU5gLgvBlEAuiIAiCIAiCYIAIREEQBEEQBMEAEYiCIAhCrCG8eEdB+Brw9PRUT5PiM8anTZtmUh/G2P7xxx8qFrdNmzZhnsxkahtLIwJREARBiDXUqlVLlbQxVhtREGIz//33H/Lly6cS2ZjgZ8qjNfkEqzJlyqhHdvJRloy75WsmnUWlTUwgSSqCIAiCIAhm8u7dO1VdgM9EL1GihHpcZGRWRL7/+++/48mTJyrDn9SvX1/V/NQEoCltYgKxIAqCIAiCIJhJwoQJlTiMCixTxbJhmvAjLC125MgRVdfU1DYxgQhEQRAEQRAEI/j6+qqnEukvpj5T3RRYYzZ0fVS+ZgVC7RnlprSJCeRJKoLZbLXLbvVnMfu1HV96FwRBUFaNQKs/DwFWfmvOmjn9V3NfOjWgKYYOHWqwjg8TGDJkiEXG9/PzC/MkJi3+lu+Z2iYmsO5voSAIgiAI3xw2djYWGad///7o1auXwbrQTw4yBz55ihnKoTOhtfdMbRMTiEAUBEEQBEEwgoODg0UFobFnjh89ejRMNrSbm5t6/KWpbWICiUEUBEEQBMGqiBPXxiKLpTl9+jQqV66snptOfv75Z1y4cAFbt25Vr58/f465c+eq9TY2Nia3iQnEgigIgiAIglVhY/f57V8+Pj6qkDW5evUqHj16pGohZsyYUQk68uLFC+zdu1eXfcxyOCys3aBBA+TIkQN3795F8eLFMXz4cN24prSJCaQOomA2kqQiCMLnQpJUvn4+R5LKzsS5LTJONc/LJrcNCAjA/v37w6xPkCCBEnla7ODZs2dRqlQpg2LwFI4UlcmSJUP27MYTbExpY0lEIApmIwJREITPhQjEr5/PIRB3J89jkXGqeFzCt4q4mAVBEARBsCoslcX8LSNJKoIgCIIgCIIBYkEUBEEQBMGqiIkM5G8NEYiCIAiCIFgV4mI2H3ExC4IgCIIgCAaIBVEQBEEQBKtCXMzmIwJREARBEASrwsZWYhDNRQSiIAiCIAhWRRwRiGYjMYiCIAiCIAiCAWJBFARBEATBqrCJIy5mcxGBKAiCIAiCVWFjKw5Sc5EzKAiCIAiCIFiPQJw3bx5++OEHlChRAteuXfvSuyMIgiAIQixJUrHE8i3z1bqYDx8+jG7dumHZsmVInTo10qZN+6V36asnICAApUuXxowZM1CoUCHEKmxskLRaWaTv2AzJqpfDw8XrcbHDwM+6C8+fP8eMGdNw/tw52Nvbo2y58mjbth0cHBzM6mOpNpwkbdu2BYf+/Rfp0qXH5ClTDfYlMDAQ//57EJv/twn37t1D/PgJULRYUfz8c2skSpTIoK23tzfmzpmNI0cO4+PHjyhYqBA6deqCxIkTh3uspvSxVJu2v7TGkydPDLZfu3YdtO/QUff65o0bWLhoAW7dvAkbGxuUKFFSvR8/fnz1Ps/BrJnTcf36dbWuatVqaN6iJeLECX/ebEofS7XZuXMHNm3cgKdPn8LF1RW1atVGnTp1w/xm16xehd27d+PVq5fInScPmjVthg0bN+C/M2cQx9YWJUuURIeOHeHkFHzcxnj16hVmzpweYR9LtalZ4wf1XdSnXfsOYY6NfPjwAd26dYHH06eYPmMWMmTIAFO5ePEi5s6di3v378PV1RV169RBnTp1zOrD8338xAls3boV586dQ926ddGubVuDMTw8PLBm7VqcPn0ab9++Rbp06dCsaVMUK1YMMcUjd3fMnDEVV69egaOjIyp9Vxk/t/oFtra2ZvW5ePECtm3djGNHjyBf/gIYNnyUwRg8H3v37MK2rVvw8OEDJErkjJKlSqNFy1ZqzC+JxCB+wxZE/vhy5MiB+vXrKwuidtEXok9QUBBOnDihLmqxDefCeZChc3M8mLcar46d/ezb54Vw8J+DEBgQiDlz52PMmL9w5sxpzJg+zaw+lmrj7++H2bNmIk+evKj03XdG9+fB/fu4cOECOnbqghUrV2HYsOFKoPw9flyYtlOmTMLly5cwbvzfmDV7Lj68/4ChQweHubFHtY+l2pAuXbph67YdukVfHFJQDxjQH2nTpsP8BYswbfpMvHz1EmPGjNIJD76fImVKLFy0BL/3H4Bt27bin5Urwj0+U/pYqs3evXvU5/vTTy3UZ9W7d1+sXbMa27ZuNdinSRMnKCH5a6/fsGr1GjRq2BijR4/G61evMWPGLEyYMBE3b97AhL//jvB3P2LEsAj7WKqNxtBhIww+O2PikEydMhnp06dHVKGoHvTnnyhSpAiWL1uGjh06YMHChdi9Z49ZfS5cvIi9e/eiXt264YrV1WvWIGOGDBg7ZgyWLl2KMmXKYOiwYbh06RJiAh8fHwwc8DsSOTtj3oIlGDR4GPbt24vFi+ab1efFixdYvnQxSpQopRZjXL1yWU12evz6G1b8sxYDBv6JUydPYMZ0w8mp8HXyVQrEP/74AxMmTMDNmzeVOGzYsKFa/9dff6nXXH788UcMHz4cXl5eBn27d++OadOmYdKkSahatSq6dOmi1vPmM2fOHNSoUUOt//3339Vs2FS4PW63VKlSaNy4MVavXm3wPq0ifH/btm349ddfUaVKFfz000+4cuUK7t+/j44dO6JixYrq77Nnz8L0HTVqFCpXrqz27e+//4afn5/ufVpSOLa7u7tBv/Lly+PIkSMG2z948CB69+6txmnevLnBRat69erqb+fOnVXbJk2aILbw5vRFnKrVHs+27ENQQMBn3/6Z06dx//49dOnSFUmSJEGGjBnRokVLdVF9/fp1tPtYqo2dnT0mTpqsLFEO9sYtmuzXvXsPZM2aVVke+fq7St/h6rWrBu0ormiFbN3mF2WJTJo0KTp17ozbt27hwoXzRsc2pY+l2pjCqZMnlfWxffsOSJgwoRqHFldatm7fvq3OnbeXl7JMuri4IE+ePKhTtx42bdqoBLkxTOljqTYHDuxH8RIlULpMGcSLFw+5cuVCjZq1sHZtyHWF1w6ORXHIMfgdiGMbB56eL9Q5S5Y8OdKkSYtWrdvg2LGjePLksdHjunLlMq5dvRphH0u1iQrbt2/D8xfP0bhx0yj33bxli7IAtmzZUn3+JUuWxHeVKmH9unVm9SlYoAD+HDRIiUhapY3RrWtXdT9Injw54js5oUH9+sqKePjTtdjSHD70r7Ied+7SXe1/9uw50LBhE2zdslkJwej24bVmzF9/o3yFiohrZ9zZmCdvPrTv0AkZM2ZS15QsWbOhTNlyuHbtCr404mL+RgVi69atUbNmTeVWptAbMGCAWk+hyNdc2rdvj507dyoXgD68qPbp0wfnz59H3759dQLx559/VgKRf/k+RRtFUng/sNBQjHK748aNw3fffafc37NmzdK9zws/rXOtWrVC5syZ1bY5Q6NQ5MWkQIECSvhevXo1jDBr0KCBmol26tQJv/zyC6ZOnarOgYavr68aO/S+cp0mcrXtN2rUSJ23fv36KXdWpUqV8O7dO9Vm5MiR6m/Xrl3VsfTv3z9Kn4s1QzdM8hQp1I1PI3/+Ampicf36tWj3sVSbqMK+Dx8+xIEDB1CmTFmD93iTpzUoX758unUUa3RzXr1i/MJvSh9LtdFYtHghateqqdzNCxbMN/j+BwaFb+mkkOFYvCnqu+j5G3z//j0ePnhgtJ8pfSzVJigwCDYIK0Dovnzx4rn6//ixo8rtni9ffr19vIoECRKoG3bI2AV12zV+XJH3sVQbjXF/jUGd2jXRoUM7rF2zRol5fR48uI9ly5Yqy6ltBC7/8OB1NG/evAbr8hcooFzHH0IZDczpYwq89vIaG1NerqtXLyNjpszq3OvvN+8Ld+/esVgfU47zzp3bOHb0MMqUKYfY8CQVSyzfMl9lDCItIBQ5/MFRxGlkzJhRLRqMp0uWLJmyNLKPBmfjCxcu1L0+duwY1qxZg8ePH8PNzU2to3DKli0b/vnnHyXqIkM/Zo/bZRwHrZy0COpD0UUBRjjDzJ8/PwYNGqRrx2Nif94s+OP9999/sX37dnXxyp49u2pDlwtntxSZ7B8VKAx79Oih209nZ2ccPXoU1apVQ+HChXXnR/+8CsGxVc6JnA1OBc8drQjhWZpN6WOpNlHhjz9+V7GMpFDhwuig55oN3t5LxI1rFyZmzcXZOYJjjbyPpdqQwoWL4LvKlZUL+cb165g8eSIeP36EgQP/VO8XKVIUCxfMx/x5c9G02U/w9fVRIpLn7OXLl8Hn1Dn0OXX5tA+vlHU17DFG3sdSbcqULYuZM2bg6NEjKFSoMO7du6tzL3P/kyRJqlyiqVOnwaKFC5S1jcfm5OQURogwZtXR0SnCzy70/oTuY6k2pGKlSqhfvwGSJk2mvof87NiXcYiEImXM6NFo3boNUqZMiZufJrBRgdvLnTt3mO8Qef3qlbLsWaKPKTAekQKxUsWKiAmMfZ9omQ5+76XF+kREt64dcfdOsLCkOGze4ucojyHEPr5KgRgedLcxweL48ePKOkcrCS+ad+7cMRCIdAPrs3//fmVNq1WrlrJeEP719PQ0OTualklu+8aNGyqGj8vdu3fDtKNrQoMXP6IJM/11tBRQIJ46dUqJXk0cEoo3ugXOnDkTZYFYtGhR3f90XdGNwBuNqfDizUUf/6BA2Nl8lcZoi2ATQ30s1SY0o0aNURYbuq2nTJ6MYUOHYOSo0YbjGhmYP43w3Gqm9rFUm06dgy3/muWjY8fOKk6R4Rb8DaVIkQJDhw7H4sUL0bpVSzjEi4eGDRvh3t27ESShBEXjpJrSJ+ptqlf/XoWFLFywAH+NHYNUqVIr6//MmTN0+08rKZMI0qRJg3nzF6qwk759euP582fKmsrft8H4EXx24e5TpH2i3qZXr966/0uWKoUXni9UUlKr1q2Vm3zB/HnKAFC5cpUo7m+ke6GI6DtsiT76HDh4UCVS/tarl0qm/Fx8uo0ZtUJbso/G1Gmz4O/vj9u3b2HShHH4e/xY9O33B74kNtGwPAtWLBAZP8gfMi10qVKlgp2dHcqVK6cutPqEnmHTWkdr3vjx48OMqQm2iKAQpGhj7ErPnj3VTIzu3F69eoVpy31CqIuOsXWaUOXMU98NoMF1mms4KuhvS9ueti1TYAD80KFDDdY1tXHDT7ZJYM1QkF+4eMFg3Zs3b9S5o/szun0s1SaqxI0bF5kzZ4G9vR3OnTuLH38Ijj9lwoCrq5u62Ht5fTCw5L15+0ZnZQh7rJH3sVQbY2gJA7Qiar/ZPHnzYtz4CerYtMxoQvH45PHjMBY1nlO1Dy7hf56R9bFUG1K3bj216Ceu8PeaPHkK9TqxW2IlFtt36KCEFWHM4rq1a1TykTZ5pHDk8btG8Nlp29cI3cdSbcL77Oie5KSYcYuXr1xW1ijtO6nRpXNHZTXubeS6GvaYXMPsi/baOdzzEPU+EXHo8GEVL969WzcVX24JGtSrpQulyF+gIEaOGqv2m0YQw/0Ojk2O6NoU1T6m3Fty5MiJ5i1bYcyoEejUuSsSJjSsjvA5kSxm87Eaic2LCxMyWKKASSJly5ZVNwrebCIjS5Ysyr1MK52W5KItpmTQ7dq1SwlSJr+wLiMtlBGVyogKjFekANW32vGiRUsJ39MXvPoJObSmhrb0RYa2zxEJRrrIuX39pVGcYLe8NZMzZy5VauOZh4du3YXz59U5YzxZdPtYqk10adasufrL7GiKQ5IjZ04lRM6fDxGlDx48UG62nLlyGR3HlD6WamMMZlJqoik0WqZs167dlPua7meOxfhN/d8I45I58UoTTsksU/pYqo0xDh44oJJRmEChjRPaspUlS/A14enTkBJA588HhxOEd/5y5sqpJsn6giF0H0u1Ce+z43fZ5ZM4njZthkGG86RJU9R6lrnRtz5GRM6cOVXJGn14jjOkTx+uqzg6fSIqw8akya5duqiEQEuxdv3/sGXbLrVQHAbvd27cvXNbnXuNC+fPqRhX/XhQfaLTx1QCPn7+JEJjSJKK+ViNQOTFlVaRs2eDS6Dw4vvbb7+Z1JdJIIw9ZBygJrLonl6+fLmqdRUZnI3RTau5ainoOHO0BLVr11Y/2hEjRujEG5NZKEiZ1UyYoUmryPr163X7PmzYsChvixdpupwfPXoUbhvuC2vm6S/fgnu5cJEiSJ8+A6ZPn6bCF+iqXLJ0MSpV+k5n2eLFllaP3bt3mdzHUm1MYfPm/2Hr1i0qS55WCIZFzJ07R8Wc6ru/+H0qW7YcFi1aoMSZVoMxc5YsBgkRdN3OmT3L5D6WanP2v/+weNFCuLs/VMfBm/isWTOUq1k/dpDnixnLvBYcO3oUCxcuUNnf/L3y3Dk6OamafbScsKzOhvXrVC1FXkcI+/Lz1LKnTeljqTb8DXJ/GebCcJXFixfh0qWLujg9LYaYno85s2crbwLbHth/QFly9uzepSYUPEeMxSxZshRSpkyl68vj2rhxg/o/V67cSpjPmjkj3D6WakMr6Lp1a9W1kpZFfi4rli9T7mRjnpLoUuPHH5WVdsmSJercMM5claepX1/XhrHX3//wgzIumNrHFDjuWIrDrl1VbHdMw6xhWm9nTp+q9p+Tj9Wr/8GPNWrqwgxoWa/xQ1WcPHHc5D6msHrVSvVd43fPx8dbicwlSxaqmqNf0nooWAaboKj4F2MRY8aMwdpPxUg1Jk+erBI3mFxCiyAF1ObNm7FixQpdsVOuYxwg++vDmSMzmJnQQpcHMzwZkzhx4sQIiwMTxnNRyDGhhH1ZboazRpa60U4vxQNn/rzoaAkgvNnzhkhRyyxGbTbNmEPuBy2bZM+ePWjRooW68NMVw5vIypUrDWIpKQ7p4mY8Ei9uzPJmIs66deuU693Y9gnbU3xqiTjMxh47dqyqMcntM0knMrbahcRHxhQ2trb4wSdsFuaHW/dxIKflZujhkf3aDiWsZs6YrqwiweEL5dG2XXtdNirPceNGDdDz116oUiV4nyLrY8k2nTt1UNn3oVm7boMqWsvvBYsq09LOQHROBkqVKo0GDRuFuTnz5k3xd+ToEXz091eJEoz70/8tUCDy5q/VHzSljyXa0CtAobtr5w4lNHgczMRu2KixQXHeSxcvYv78ueo3RW8CS8mwDJAGhTZj+nhzjJ8ggXqPAlKzpFMgdu/WBaPHjNWJ08j6WKoNJ3kse7Np40a8e/dWWZF/btXaIJaaeHg8xayZM9X3wsEhnrIw1m/QEOvXrVW1MpksV6JkSXTs2MnAZU+BqF+cWhW4njE9wj6WaMPPlvvGMj5MtqG7vHKVKqoIuCaOQ8OC5z17dtcVyo6D8DPU9bmgFb2+dw9uLHpdt65B0WsKueEjRmDRwoVKaJvShx6TJk3Dlt3JmiULpkwJtnT2/PVX5eIPTckSJfDnn8FJVJEREMXoL4rxmTOmqbqEwUWvq6BV65Ci1xSI7du2xp+Dh6FY8RIm9SHNf2qsrPf68PqzYdNW3ee96p8VOHH8mPqeMvGoXPkKqFuvQYRCM2vmqNe2jCrnq1smkzr/jn/xrfLVCkQKQF5geEHUh+t4k9QC1ZnkQaFDqwFhNjBvhuE9eYVWC37peSFmRmBU4EWFrl32pZCjhUYTY7zgnzx5Uu2vdjOmsKTAZUkPbVu0dlAwFixY0ODmz7ZMmOENhOLNmAubAoXWS7rFadnj9iiWaWUytn3CbVEkUqjqixEKZF6wTUmC+RwC8UtDgSgIwpfHVIH4NRNVgfi18TkE4oUfKlhknHzbDuBb5asViELsQQSiIAifCxGIXz8iEL8OrHuaYiHootbcB6FhoghjFQVBEARBiB1IFrP5iEA0AZYoyJTJeGZXVN3QgiAIgiDEfBazYB4iEE2A8Yym1EMUBEEQBEGwBkQgCoIgCIJgVYiL2XxEIAqCIAiCYFXIo/bMx/orHAuCIAiCIAhRQiyIgiAIgiBYFeJiNh8RiIIgCIIgWBUiEM1HBKIgCIIgCFaFCETzEYEoCIIgCIJgQd6/f6/qJBt7LG7odj4+PmHWs5+bm5vu+fN8Dnho+Ahh/WdnWxpJUhEEQRAEweqymC2xRJUdO3aoJ6xR3Dk7O+P3339HRE807t+/P3LkyGGwsO5y/vz5dW0OHjyIpEmThml39+5dxCQiEAVBEARBsLonqVhiiQq3b99GnTp10L59e3z48AF79+7FrFmzMGHChHD7TJ06FS9evNAtjx49QqJEidC0adMwbfXbccmSJQtiEhGIgiAIgiAIZjJnzhykSJEC/fr1g52dHYoVK4a2bdtiypQpJo+xYcMGvHz5UvULDV3NxtzRMYUIREEQBEEQrC5JxRJLVDh+/DjKlCljsK58+fJ48OABnj59atIY8+bNU32yZcsW5j26rGldzJo1K5YsWYKYRgSiIAiCIAhWhaViEH19ffH27VuDheuM8fz5cxUrqI/2mu9Fxr1797Bv3z60a9fOYH3ChAkxf/585VZmUkvv3r3RqlUrrF27FjGJCERBEARBEAQjjB49Wlnu9BeuM4aNjQ0CAgIM1n38+DFYbJmQ8LJgwQK4uLigfv36BuuLFy+ONm3aqKxoe3t7dOjQAfXq1cOMGTNi9DOTMjeCIAiCIFgVlqqD2L9/f/Tq1ctgnYODg9G2qVKlgoeHh8E67TUzkyMiMDAQixYtQosWLRAvXrxI94tu5pUrVyImEYEoCIIgCIJVYSmB6ODgEK4gDE25cuVUVjKtiFp9wl27diFXrlxhahqGrmHIdg8fPgzjXiYsk0PrpD6nTp1ChgwZEJOIi1kQBEEQBMFMOnTooCyB/Hv9+nWVSLJw4UIMGDBA14albxiXePPmTYO+jDEsUaIE8uTJE2bcjh07Yvbs2bh27RquXr2KX3/9Ffv371fZ0jGJCERBEARBEKyKL1EoO1myZEq4PX78GJUqVcLEiROVsGvWrJmuDWMIEydOjLhxQxy4tCgePXoUnTp1Mjru8OHDlThs0KABateurZJZjh07hu+//x4xiU1QRCW+BcEEttplt/rzlP3aji+9C4IgKKtGoNWfhwArj/7Kmjl9jG/jYWfDRI/oknbGOnyrWPe3UPgsfAvi6XqO6rB2RlefA2tmyNmusHZGlLDuz5DMG5sC1k4cG8NMWCHqROcxeYIhcgYFQRAEQRAEA8SCKAiCIAiCdREq61eIOiIQBUEQBEGwKixV5uZbRlzMgiAIgiAIggFiQRQEQRAEwaqQJBXzEYEoCIIgCIJVIS5m8xEXsyAIgiAIgmCAWBAFQRAEQbAqxMVsPiIQBUEQBEGwKsTFbD7iYhYEQRAEQRAMEAuiIAiCIAhWhVgQzUcEoiAIgiAI1oU8i9lsRCAKgiAIgmBV2Mij9sxGYhAFQRAEQRAEA8SCKAiCIAiCVSFlbsxHBKIgCIIgCFaFJKmYj7iYBUEQBEEQBAPEgigIgiAIgnUhWcxfhwWxR48eqF+/vsntf/nlF/z888/4ksSNGxf79u2zum27uLhg27ZtiClevHiO+/fv4ePHjxbtY7k2L3Dzxo0I23h4PMXTp08REBCAL0Fcl0RwLpwHcRMlwNeIo6MtMqWPj0QJTZ9/JowfFxnTOam+4bZJENzG1cUOXwIbu7iIny0zHFIkM7lPHEdHxM+aCXauLhG2s3NxhlOWjLHqppY2VTykSu5gcvu4cW1Un6SJ7SNtmyKZA7Jnjg97OxtYgmceHnjw4H6UfrOm9LFEG67nNeXhw4fw9fUNf5xnz9S1KTAwMNw2vG49uH8fz58/D7dNdPqY0sbLywsPHjyAj49PuG2CgoJw9+5dPHr0CLHBxWyJ5Vvms1gQ+QOJyg83qu1jAm4/oh/q59o2/3dwcMDevXtRvnx5s8fmhSAmjuvdu3cYPWokrl27ioQJE8LPzx+/9uqFYsWKm9XHUm0uXDiPDevX4/LlS/jw4QMWL1mKJEmSGuzPnj27sWL5MsSJE0eNERDwEa3b/ILKlavgc5AgZ2Zk+u0XJKteHg7Jk+BU7Q54tu0Aviaa1UuLX5qlx4uXfkiaxAFbdz/FhFk3ERRkvL1LIjv075kdBfO44NkLHyRL7ICVG9yx8J/7ujYUjX/2yoEiBVzxxMMHyZI44N5DLwz+6wo8nod/w7UkSatXQs5xQxHo5Q07Nxe8OnEGFzv2RsD7D0bb29jaIvvw/kjZsBZ83J8oUfl89wFc+W0Qgvw/GgjDnOOHwa1sCfg8fARbJ0dcGzASnvsP40uRJaMThv6WHY7x4sAubhw88/TFn+Nu4NHT8IVB41op0bJBGrx87a8mBvceemPw+Ot4/TbsRCxlMgfMHZcPCeLHRetfz+HuQ+9o7+vLly8xcuRw3L93H05Ojkqg9OnbD/ny5Terj6XanDx5AtOnTVXvOTo6qglqvXr18VPzFro2p0+fxsYN63H9+jUlwtau2wAnp3hh9pvt/h4/HnZ2dnj//j2yZcuGAQMHqmteeJjSJ7I2FHyLFi7E5cuXkThxYiVkK1SsiC5duihjBvH398fm//0P27Zvx0tPT+TOnRvDR4ww8VMUYiufZbo6ZcoUrF+//nNsyirgj+27775T//PCQpHIv7GZmTOm4+3bN1i2fAUWL1mGOnXrYuyY0Xj50tOsPpZqc+3aNVSrXh1/DBgY7v7wgj/+7wmYN38hlixdhsZNmmLypIlq9v85cC6UBy8PncahInXwNVK0oCvat8yIfsMvoXH7k2jT4wyqlE+G+jVSh9unV6csSJ7EAfXbHEfzzqfRpud/aFgrNb4rGyLeWzRIh5zZEqJxuxNo2fU06rY6Bru4NujeNvNnOS6HVCmQZ9pY3J00G4eLV8WhopXhmDY1sg/pF26f9J1aI3mt6jhetQGOVayNwyWrIUGOrMjcu2tIozhxkH/RNNi7ueBw8So4XrkeTv7YFPZJEuNLQSvg8N7ZceHKW9RrewZ12pzCc08//Plr1nD7FC/ogvY/pcfgv2+gRfdzqN/uDF6/9ccf3cL2sbW1wZ+/ZsO2fc8ssr/8fZLlK1ZiydLlqFixEkaNHKGEjjl9LNGG1+2/xo5FmTJl1XVp9px56N9/AFasWI5LFy/qxrlx/Rrq1quHX3v9Fu4+v3r1CqNHjUKt2rWxZOlSLFu+XE2Mp06dalYfU9pQINaoWRNr1q7F7DlzMG36dBw/dgwrV6zQtfH68AGeL19iyJAhKFmyJGIDNjZxLLJ8y5h99HQHN2jQAJ07d0b69Onh5uaGTp06GZjSf/31VzRs2NCg3/Lly1GwYEEkSJAAhQoVitDtefLkSSRPnhxjx47F/fv31azl9u3bBm04jjYGLURsQ2FapkwZ5VbNmTMntmzZEu3j5Hbr1aunxuIxNm3aVLkhNbRtTp8+HRUqVFBtcuTIgXXr1hmMc/XqVWUJ5OwsT548mDdvXhi3b7x48bB//371f4oUKdRfCkaOzz6mnANy48YNVKxY0WBboXn9+jXat2+vtsN9rly5Ms6ePRulc8NjP3z4EOrWqw8np/hqHc8VLXEHDxyIdh9LtSGNGjVGiRIl1frwYBs3t5Cbc5EiRZW1lbP+z8Gj5Zvgvng9ArzDt9TEZmpVTYnzl17j9PnX6jWtfHsPPUONKsHfYWMULeCKbXuf4t37YEuT+xNvHD3liTrfp9K1Sexmj/sPvZR1inj7BOLKjXdI7Ga6+9McUtaviYAPXni4MPiG+PH1WzyYtwzJ63yPOPHCWnqIW7mSymLoffeBrs+TNf9Dqmb1eedS65JWLg+XIgVwpe8Q9T7xf/UaT9ZswpeiSD5npEweD4tWP1SvAwKBJWvdkT1zAmTNGN94n/zOuPvQC6fPv1GvP34MwsYdT1GsoAtSJDX8jNr/lA7uT72x97D5vyn+Lk+fPoWGDRspLwvhpM7b2wdHjx6Jdh9LtfH29oa3txfy5sunK9qcL39+9T8noxrNfmqOwoWLwAbhuzMPHjyo/jb4FKrl5OSEevXr4+iRI0rQRbePKW0qVaqEokWL6o4hderU6vVFPZHr7OKCdu3aIU2aNIg10D1sieUbxmyByFkSRRDN0xRyO3bswPbt2zFgwIBwXcYUKvwydevWTYmdhQsXKsFojD179qBKlSoYPnw4+vXrF65FTd91qrXhbGbQoEFKSLVu3Rp169bFzZs3o3WMNWrUgJ+fH86fP49Tp07h8ePHSohoaNscP348Ro4cqWZdFM/NmzfHkydPdPtYs2ZNJXZp0VqzZg1mzZqFN2/eGLh99V3MWizHrl27VOwHt2/KOeD7tWrVQpIkSXTbmjFjhhJU+tupXr26mvEeO3ZMnaeqVasqMerh4WHy+eGxcqysWbPp1tnZ2SNjxky4dftWtPtYqk1UePPmtYoDottl2rQpKFiwEHLkyBnlcb5FsmVOgGs3DW9WV66/Q8Z08ZXFzxjvPwQgsYthzJqri70SJBprNz9SYzStmwZ5cybCj1VSoHzJJFi8KsQNHZMkypsT7y5eBfR+o2/PXoRtvHiIny2T0T4f376HQ9IkBuvskyaGvasLHNOl1onID3fuwevWXTimS4N4aVPrxOOXImum+Hj52g8eL/x0667deo/AwCBky2RcIH7wCoBzQjuD8ElX5+A4Uf0+xQq4oFxxN0yae9ci+3r70++bLlGN+PHjI02a1Lh961a0+1iqDSfsdevWw/Jly3DyxAkV5jJp4gRky5YdxUuUiNqx3rqFDBkzws4+5LeSPXt2db3nNTC6faIzrhZnyPuYYN1YJAaRM4q///5bWbX4pRk1apQSRyNGjFDWsNAMHToUvXr1Qps2bdRrxjUYE4gUNRxn/vz5YSyQpvDbb7+hWrVq6v++ffti48aNyqoYkVneGLQ80hrHxJGkSYNdXwsWLEDWrFnVDEw/NpCitHTp0ur/3r17q2OloKRY+9///qeEJcUHrYY8b7Q4lojgYmFra6v7q8V7mMLmzZtVQDGFn6urq9rWtGnTdPtGKOQpHg8dOqQEvnaeVq5cqUQ/rcKm8O5tsPUjUaJEBusTOSfCu7fvot3HUm2iwoXzF7B27RrlnuY579HzV91nIERMooR2ePPOMOaMrka6FRMkiItXnyyA+qzf+ghtmmWAxwtf3L73AcULuSJ39kRwcoqrRKX/xyDcvvcem3Y+QeumGeDxPDgGcf/h5zh7MdhSGdMwwcTXwzB43+/lK917xnBf8g8KLpuFzH27wfPgUSTImU1ZIlUfN1d433eHQ/Kk+Pj6DQqumAPH9GkRxyH4N3i171B47juEL4FzAju8DfUZUhdTyIeXdLTzwHM0qpkK/btlwdY9z5DY1R4/N0yDgIAgOCcK7uPmYod+XTJjyIQbSlBagreffvsJE4b67Sdyxtt3b6Pdx1JtSLXq36vYvSlTJqkYRMYYduveQ2d1NPlY371DolCxhs7Ozgb7Ep0+0Rl37dq16t7S67fwXeKxASmUbT4WcbDTRawvXooXL66sXcZmIAxwdXd3jzThgsKLbtxVq1ZFSxwSmsH14X5duXIlyuPwB54pUyadOCSZM2dGsmTJwoynP6OkSZ5CkHEemns5S5Ysap3+uYvI7RlduF/cFsWh/vnQ3xaFKi8CdEHzgsXF3t5eWSnv3LljdFyGDrCP/hL4yZLJ2El9/Hz9YBvXuLjS1kfUx1JtokLZcuUwecpULF22Ak2aNsOfgwbi1q2oW52tGXv7OMieJYFuoQuYfAwIhF2orFQH+zg6t6MxmJAyZsp1FM7ngg4tMyJu3DhY9A8zQoOUOCSdW2dGtQrJ0LTDSbTochp1Wh1HmlSOGNovFz4Hgf4fEcfeMHOa1kOin3Ciz8t/j+FMwzbKMpilX3ckypsL1waODB7PJzj8JiggEM6F8uPZtt04Wvp7HC5SGU/Wbkbe6X9FmvVsCdKncVSZxFyYHU4+BgTBzi7s9cje3ibcz/DJM1906HcBPj6BaN04LcqVcMOY6bfUxMDPL7jPr+0y4dL1d/DzC1TbS5faUa3PkNYpjBvaGB/9nsPP665aaOGn10A/QUIfPz9fxLU1LmZN6WOpNnTR9u3zG/Lnz6+uJ3PnLUDvPn1VnOLZ//6L9JgN9tvWNsy2tDCu8AwHpvSJ6rh79+zB0iVLlDjMmDEjYjOSxRxLLIihH4qtvY4oWzayB2nTOkfBsnTpUuX2jMyKYyyJw9h+RTeD19j+cl3o7Rprp98mvHNlCfS3w/8jG5sXWQpdCtfQhCdaR48erayi+rRq1Vr99fT0NBDRtMLlzp3H6DjJkiaLtI+l2kSX6tW/x7KlS3D61CkUNGsk64JJJX06h0yE/rfziVqePfdFklBxgUkS28PL66MuxtAYew89V4tGv27Z8OCRl+51uZKJsefgc3i+CnZ7ensHYMuup/ijZ3ZVJsXPP2YTuHweP0H8rIYJMQ4pkureC4/XJ86oRSNNq6YI9POH16e4RB/3xwjw8cGj5Wt1bRjnmLFrWyTKnxueB4zH0VmK5vVTI12qYKHGrPMBY68rS66bc7C7WLtUJohvi3gOtnj2IvyM8YePffD37JBJZd4cwVape+7Bn6OPX4DaVq/2mQwmDsx8PnDME4vXuEe4r96vjsD33Tn1/7Rpdhgz9i8k/fTb52+dsXMajO/Ln7+A0XFM6WOpNpcuXVIT6Dp16+muxQxZyZAhA44eO4qChQpFeMwG+50smZq468Ns4eB9SRrtPlEZd/++fZg8ebLKKbBERY0Y5xtPMLEEFjmD586dM4gxpEuV4o5Wt9DQ6pYqVSocPhxxGQe2OXDggIprbNGihW58zSLG5AoN1pdifGBoaCHTh/vFZJWowj60qOkHFtPEzji9qIzH2I5bt24ZmO557iISrRRqXPTbmHIOuF/cFuMbNf777z+DcQoUKKCOS0t60V/CE4j9+/dXY+ovvX7rrfbp1KmTBrXB7t27h/wFQi7U7u4P8eTJY/V/uvTpI+1jqTam4O/vF0bsM16T8ZkJIigj8S3y8LE32vb6T7dQHBImpxQv7GoQi1a6aGKcuRDyPaWbklZHrU3o2ES+X6FUEuzYFxID+/btRyR2NbTg0Wrp7RMQ4+KQvDx8Aony5YJdYjfduiSVK8D74SN43wtO5ogTzwEJ8+WCbfxgsWAT2voSJw5SN6mrrIWBn+rIef57FHHs7WHnHOKm1OIWmawS04ycfAsd+l1UC8UhOXvxjSorlD9XyD6VKuKKjx8Dcf5KyHWLVkCXT+5jY59jzSrJVZLS9dsfwmyLy+hpwXF6wybeiFQckoTJ6yBJliFqoYWf7lp6axwdnXDqZMhv/+7dO8pLlb9AyJSONQq1SgSm9LFUm0SJgq8br16F3Dd4H+M1M7RbNzJ4rWadQv3Y8BMnTih3MAUn8ffzU6FQWpy5KX1MaUN4L544cSJ6/vorKlaqFKV9F75xgUixROFA4XPhwgWVoMLMWP6IjcG2jFmk+5hf5uvXr6tYw9AwI4rZvPzCtmzZUv24+MVljSV+WXnzpru6Y8eORrczYcIE/Pvvv2objD1kPB4TY6IKE0v4Y+Ex8QLAxJG2bduicOHCKmPZVBiHyFlZ165dlcWLwoxFxCOCQo3ngUJZE3emnAPuMzOTWauK26K7v3v37gZtateurdwfjRs3VpnLdC3QmsgZIuMSjUHhz3g//YWfc4sWLbFu7Rps2fw/JdZGjRoRHIxdPCS+ctrUKZg3d67uuCLrY6k2hOeArqlH7sE3I54PvtYy9Z4+9UCf3r9h9+5dauZ/6N9/MWjgALglTowKFSricxbITpQ/eNLhlCW9eh0v9dcRDL56kzscHVizMKeqWdj1l8zIk9NZuYw1ShdLjPkTC8PpU0Hs4oXd0L97NtW+XInEmDQ8H+7c98KqTSGiYcO2x6hWMTmaN0iLfLmcUbt6SjRvkA4btwVPNmKaZ1t24cP1W8g/dyISVyyDtL80R9rWTXF73DRdG6dMGVB82yolJEm8VClUDGKSKhXgVr4UCiycirjOiXBj+HgDNzTFZ945E1TCCtvmmjAcr46dwtsLUQ+FsQT33L2x+9/n6Nsps0ooqVw2Cbr8nAFrtz41qGk4e2w+9Z7GhMG58F2ZJEpY/touI8oUc8PYGYZVFiwNw2GaNmuGZcuWYueO7Thx4jjG/TUWBQoUVNc1jb/+Govly5eZ3MdSbbJnz4EsWbLi7/HjcOzoUZw/dw7jx/+lrtmVvqscpkD2k6fBEy0mC+oLPVKsWDHkyJkTo0aOVPcCxsUzFrBFy5Y679rzFy/Qs0cPdf0ytY8pbY4ePYrx48apRE3ei7hvXDgJ14fGhhufrqmMteT/NFJ8KcTFHEtczBQjrMBOtzC/HIwZHDNmTLjtKZCYFMEMY2b50rLGLGVjpE2bVs1eKMT4dJXFixcrtzOzoJmhy9I6ffr0UYWkQ8P1FDuMIaRYWrFiRbQsiLSobd26Ve13unTplCih25uJKlFxEfOismnTJrXv3B+ORdGmWVzDg5nRTHjh+eK54gUgsnPA88ttUchq2+rZs6c6F/rHxSzx33//XSXz0CLJmTGFcESJM8ZgMDbLzOzZuxveXl6qWGyjxk0MQgNSp0mjZt1R6WOpNsePH8OunTvV/7xoL1u6VP1PcVmkaFH1PevatRu2bNmMPbt3qwzEUqVK44cff9CVz4lpnAvkQo4xfdT/r89cQupmtdTC8jf3pi5BbIcu4I59z6Jlw3Ro0zQ9nnv6olv/c7h+O6R23Ju3/irTmTGG5PAJTzjGs0WTOmmUVXHn/mfYsO2RQbzb1j1PVWZttQrJUbJIYrx644fJc29h537TM+3NIejjR5xp0hYZu7VHxm7t4P/2HS526o3nO0KedhTo7YO35y/j4/tgl6r3A3fcnTYP6X75SRXDfn36HC73HBDGMni+TXek79QKGbu3V+WNPDZtx4P5yxkjgi8FhV2jGilR74cU6nNatNodG3ca1gLlZ6qVHdL60GWdKnk8JfDb9bkQYWFtliriGL5+5hXtr1+/gZowHzx4AL4+vihZshQaNmps0CZ9uvRIkTxFlPpYog2vr6NGj8GmjRuwffs2+Pn7IW2atJg0eapKGtQ49O9BZcjQrk1z58wGK950aN8eufMEh8nwnjN82DBlVFmzerWalDPRs7yegcLezg5ZsmZFgvjxTe5jShsaMjJlzoxLly+rRSNpkiQY9OefutcLFyzAGz3vGJMieR+iMeiLEIueSvS1YhNkZgXmVq1aqfIqy5YFz9CMoVm+TE3GMNZeK+0SXkAu32N7CjbO0Jh4QYthVIWOBo+JAiMqApB9Qu+f/n4Zgxa7XLlyqdkWBXZE2+Z54XkILx4zsm2Z2iaq3LptmbIVsZnrOarD2hldfQ6smSFn9YpUWykjSlj3Z0jmjQ2/rqa1YGMTux+MYC6ZjYSfWZq3k3pZZJxEPSfgW+WzPGovqlm6xtpT0ERU5sXSpUiiUlImoj6h94tub2YuU7hyZkZrHSvPa+Iwom1Hdh5NOQdSskUQBEGwdixpBPlW+SZtsBRpoZMytIVBuzHJDz/8oOpEMhaRmWCMbWR9RkEQBEEQLAQNKpZYvmHMtiAyDi+2wfgx1nYKzxLHuMTQCRufa9bBx+/xqSiCIAiCIAhWKxBjoshzTLuII3NXC4IgCILw9cIsZsE8RCUJgiAIgmBdfMFC2UePHlUl/1j3+fvvvw+35J9Ww5iPtg0Nq7boPwktquNagthp/hMEQRAEQfjK+OWXX1TpvyNHjmDIkCHIly8fHj8Ov2Yr60Az7I11JfWX0I9AjOq4lkAsiIIgCIIgWBdfwMW8ZcsWLFq0SD14ggKOD58oXrw4+vXrp2oXR8SkSZNiZFxzEAuiIAiCIAhWhY1NHIssUWH16tWqbB1FHOEDMNq0aYP169er+sYRsXz5cpX0yyfHWXJccxCBKAiCIAiC9VkQLbD4+vqqxwjrL1xnjCtXrqhKJfrwNR89yLrH4cEn8vDJZ3yyWfXq1dWTzdjH3HHNRQSiIAiCIAiCEUaPHq0EnP7Cdcbgo4ZdXFwM1mmJJnzPGHxIxs2bN5WVkI8D5qN0//vvPwwdOtSscS2BCERBEARBEKwKGz5S1gJL//798ebNG4OF64zh5OQURrDR4qi9Z4z06dOrB2do8DndjRs3xu7du80a1xJIkoogCIIgCNaFhR564eDgoBZTyJYtG27fvm2wjq/t7OzUU9NMxd7eXglRS48bVcSCKAiCIAiCYCa1a9fGv//+iwcPHqjXgYGBWLZsmapZSNFHGDPIjOVXr16p11evXjUYg5bCDRs2oEyZMlEaNyYQC6IgCIIgCNbFF3jKW5MmTbB48WJUrFgRzZo1w6lTp1SCCWsXalAQsu4hk1EYR8j2LIBdrlw5BAUFYeXKlYgfPz5GjRoVpXFjArEgCoIgCIJgfS5mSyxRfPTwtm3bMHz4cPj5+aFKlSoq6SR79uy6NnQJ9+jRA25ubur1mDFjMGHCBCRIkEC5sseNG4fz58+rWMSojBsTiAVREARBEATBAtja2iorHxdjsDxN6KLYRYoUUYs548YEIhAFQRAEQbAqmIEsmIcIREEQBEEQrIsoPgVFCIsIREEQBEEQrIsv8Cxma0MktiAIgiAIgmCAWBAFQRAEQbAqbMTFbDYiEAVBEARBsC7ExWw2IhAFwQRGV59j9eep/472sGZG1F4Ea2fAoXawfjbD2gkKkvg54csjAlEQBEEQBOtCXMxmIwJREARBEATrIopPQRHCIlnMgiAIgiAIggFiQRQEQRAEwbqQJ6mYjQhEQRAEQRCsC4lBNBtxMQuCIAiCIAgGiAVREARBEATrQuogmo0IREEQBEEQrAtxMZuNCERBEARBEKwLKXNjNhKDKAiCIAiCIBggFkRBEARBEKwLKXNjNiIQBUEQBEGwLsTFbDbiYhYEQRAEQRAMEAuiIAiCIAjWhWQxm40IREEQBEEQrAuJQTQbcTELgiAIgiAIBogFURAEQRAE60KSVKzXgvjkyRM8ePDA5PaPHj2Cu7s7viSXLl3Chw8fYs04xrh69SrevXsXI2MLgiAIQqyJQbTE8g0Tay2II0eOVIJv48aNJrUfMGAAPn78iGXLluFLkTdvXuzevRuVK1eOkhjMmDEj4sePH+44xtpEl6JFi+Kff/5BjRo1EJPcuXMHt2/fgrOzMwoUKAh7e3uL9DG3zcuXL3H82LEwfcqULYtEiRLpXgcEBODKlcvw8PBAkiRJEBTkChsLXCySJ3VAgTzO8PMLxOlzr/Huw8dI+2TPnADp0zrhw4cAnL7wCr6+gWHaZM2UABnSOuHjx0Bcu/UeTzx8EFuJ42CPZDUqwc41ER7OW42vDXv7OCic1xmJEsTFjTvvcfehd6R93FzskCtbQjjGi4OL197h6TPfcNumS+2IArkT4erN97h5N2YmikaxsYFr6aJwTJMKXnfu4/XJs5F2sU0QH67FC8EusSveXb6O95evG21n5+YC11JFgaAgvDx0Ah/fWm6Sysn0ubNn4ePrg1y5ciNlypQW6WOpNuT+/Xu4fesWEidOgrz58iFOOPFx169fV+3yFyiA1KlTh7v//v5+OHfuHF6/fo1MmTIjc+bMkR6zKX1MaePh8RQ3rt9A3LhxkT1Hdri5JQ7T5s2bN7h69Qp8fHyQIUNGZMiQIdL9E2IfsVYgpkqVCra2trBmKGgpBvfv348KFSro1ufOnRsJEiSIsE1sZt7cOdi5cwcKFiykrMAUW2PG/oXEiROb1ccSbZ48eYzp06eiWrXqBhfpYsWLGYjIQQP/gJ+fH3LkyImbN2/i9fM4cE7dE3FsnaJ9Xn6snAK/dsyCMxdeI2H8uOjdORv6DL2IKzeM3yzjxrXBqD9yI0eWhDh17hVSJo+HXp2yoOfAC3j4OFiU8BCG9cuFQnldcPzMSzjGs8XAX3NgyZoHWLzKdAv85yJL/05I36Ep/F69RoIcmb86gZgsiT0mDc0NH59APHzijR5tM2LdtieYv/JhuH1+qJQM3dpkwLnLb+HtE4CebTNi3sqH2LD9aZi2FJAj+mZHimQOWLrW/bMJxDiO8VBo5SzES5sab86cR5YBPdXfC7/0QlBAgNE+FHz55o6H150H8H7wCFn//A3Pt+3F1b7DDNql/qk+sg3vh9cn/oOf5ytk7tsVFzv3C1dMRgVOBAcOGIBUqVIikbMzpk+birZt2+OHH380q4+l2vj7+2PSpAk4deoUChcuDB9vH6xcuRwjRo5WAkufFy+eY9TI4Xjx4gV+690nXIHI61P/3/sqQZ8+fXrMmT0LVapURfsOHcM9ZlP6RNaG19KxY0fj9q3byJIlixJ/48aNxS9t2+HHH0MMDuvWrsG2bduUuLSxsVHnpUiRoujdp+/nvaeLi/nrFIh0BwcFBSFNmjR49eqVcnmmS5fOoE3r1q3Vjys0/FLSspg2bVo4ODhEuB0KBG9vb2TKlEnNzLJly2ZgTbpy5YoaJ2HChAgMDFSv+aXmuOxLUcH3ogu3yWOgENG2o8+1a9fU37t37yorFbebNWtWZeGjxTC8NpyNRXY8Gtw+j4XnOqLz9fTpUyWI2J8/6uhy4cJ5bNiwHuPGT0CuXLnUjPS333ph3rw56Nevf7T7WKqNRsdOncO1ai5auED9nTFzNuzs7NSFsV7D7vDy/B8SJGsSrfOS2NUevTpmwdT5t7Fx+xO1blCvHOjfIztadDlttE+9H1Mhb05nNO98Cp6v/NS6Pl2yol+3bOja/7x6XSifKyqUSoqWXU/jzv1gMVG7ekr06pgVazc/wgcv4zf3L4XXPXf8W7g2kv9YAXlnj8DXRvc2GeH50g89h1xBQEAQCudzxt9/5sLR06+Uxc+Y5bBnu4yYtvAe/rfLQ63LlS0Bpg7Pg1PnXsP9iaGlt1eHTDh43BPVKyTF5yRDlzZwzJAWxyvVh//L14iXLjVK7l+PVM3q4dHSNUb75J4yEs8oCPsEC0L7ZElQ6vD/4HngKJ5t26PWuRQvhJzjB+NCu9/wbMtutY7WRvskbhbZ74kTJyB//vz4vf8f6vWOHdsxc8YMFC1WDEmTJo12H0u1WbJkEc6fO4cZM2YiSZLgdbSs8f6nD68xf40di0aNm2DG9GkRHvP8+XPhEC8e/v57Auzs7NV4vX/rhWLFiyuPSXT7RNYmMDAA5cqVR//+A3T3CB4zBWDx4sV1x5chY0bMnjNXJ4Dv3buHLp07otJ336Fo0ZCJeIwjWcxm80Uc7HQHN2/eHKVKlVLWsuzZs6NkyZJ49uyZgYu5e/fuute0pPXt2xdubm6oWLEiUqRIgbFjx4a7jeXLlyNPnjxKSFGQ0goXOqaxUKFCOHjwoPrfy8tLtenTp4/6cZctW1YJsjFjxkT7ODt06IAmTZqgfv36yvVQtWpV5bLUaNeunfo7bNgw1e63335Tr7kfxz65QY21MeV4yJ49e5QwLFGihDpfXbt2VUJYHwrQYsWKKUHFY06ePDnWrDF+QzCFfw8eVCKb4xFeaKpX+x7Hjh5Vgi26fSzVRuPYsaPYv3+fsgKE5tatm+qCSHFIOOu1j58LPm9PRPu8lCmeGIFBwLa9IZ//+q2PkDFdfGTJYDx0IHvmhLhx+51OHKr9Pv0SBfK4KFe14tON5s3bkMnUqzf+CAwMipUT6McrN8Pf8xW+RmjdK1HIBZv3eChxSM5ceIMHj7xRqbRx63jGtE6wt4uD4/+FHPOVG+/h5R2AiqH6fF8pKdKmdMSi1Z8/ljpFnerw2LRDiUPi8+ARXuz5Fynqfh+uy9gxbSrVRsPv2Qu8u3QNKeqHWNDSd26lXNWaOCT8/D9cv232Pj98+BB379xBjRo1deu++64y7O3tcPTIkWj3sVQbGjO2btmCevUb6MQTyZkzl+7aorFixXI1sa9e3fj51r8P8npWtWo1dX3TxsuSJau6/kW3jyltuL5MmbIGBoRChQqre8oj90e6dYULFzGwjrq6MjzHJsy9J6YJsrGxyPIt88UiMClkaCV8/Pixsl7xy9OzZ89w2w8ZMgRLlizB4cOH1Y+TSSyhZ2EaU6ZMUWJoy5YtqFWrVpT2a/v27Th//rzaxubNm5WYPXr0KKLDgQMHVPwgE0N4jLwo/P7777r3Dx06pP4uXLhQtfvf//4XZgxT2hjj/fv3aNasGdq2bYvnz58rYapZVDUoiqtVq4batWsrt8b9+/exdOlS/PzzzzrLZVR58OA+0oayBtM6TEsmP7Po9rFUG0JL6v59+9SF/Pd+ffHnoIHqYq6RNFkyZbHV56OvO4ICPyDgY/ANNKpQCDIukLGHGncfeAW/l964QPR47oN0aSgwQi5SWTIm0I1HTp9/rSyFowfmRtO6adC6SXq0b5EB42fexPsPsct6+LWTNpUj4saNg/vuhjGH99y9VPynMZ4+D441zKz3GSdPYo/4TrbIlC6kT7pU8dCuWTqMmHxTJz4/FzZx48IpU3p8uHHHYD1FXILsxmPb/F+9wcd375EgV3aD2FKOkyBnVt06l6IFVcwhrZMpG9VC4kplYBs/+mEa+jy4f1/91f/N8xrLyfj9B/ej3cdSbTj59PX1Rf78BZR3Y+/ePbhy+XKY+9bFixewe9cudO8R/v1P4+nTJ2rM0B43vg7vmE3pE51xycmTJ9QEOn2oGEPGKW7buhVr1qzGoEEDlKv6s1oPha87BjFHjhw66xiTCSgAmTgxe/bsMK5Yisfp06fjzz//VFYyEi9ePAOxpcE2HGPfvn0oWNC4uT0ievXqpaxuhBa/6tWrY+7cucraGR04M6MIfvv2rRJj48aNw+eAyT10GfN8ELpTaXGl6NXYsGGD2q969eopSyuhi5lubsaQ8DOKKhSdCeIHixgNLZ6S70W3j6XaJE2aDLPnzNO5gBj306N7NyxbugRt27VX65o1a65icUYMH6Ys3Ddu3kCA72P1XlAgxYFLlM8LBcH7UAkptCJ9DAiCk6PxuJw1mx+heqUUmDKqAA4cea5iEJngoo2n8fK1H1yc7ZTFMV68OLCNYxNmW4L5aJ9TaOH97n0AkrgaD1d49NQHW/Z4oH/XLNi08ym8fQNQtVxSPPP0Q3zH4MuvXVwb/PlrNsz/52EYl/PngILNxtYW/m/eGqzna9tPv58wBAXh1qjJyDq4N+yTJlYWx2Q1qiDAyxtxE4b0sU/sCpfiBZGy/o94feYCEmTNBIdUyXGuZVe8PXvJrP3WftPab1wjQYKE8I7gWhNZH0u1efvpfC5ZvAjv379DipQpceH8eSUihw0fqe5hTOYYP+4v9OjZU90H6Wo25ZjjG7nORXbMEfWJzrj37t5V4TgNGzWGi4vhNfHDBy/cuXNbHd+rly/hXLBQuAadGOMbz0D+qgWi5gbU4I2YQpAZqIzr0IfWL2ZVRSb4KIqYVXby5MloicPw9ktz90aVwYMH4++//1bZsfwB0XpH9/Dn4Pbt27p4Sg268vWDhGnZ5KyxYcOGRoWtMdieC61ymjXz1avXSJ8ho/rc7O0dDKyU+hef8GIgTeljqTbJkiUzeJ+un4oVK+HMmdM6gcjzNGfuPBw8cEBZX2kBOHcjB94+mQ2bOI6IjCrlk+nEBF2/B46+gK9fYBghGM8hDuLa2hjNSiavXvur+MNqFZMjTSpHPH3mg2F/P8HiqUV0sYWVyyXFz43T4+eup+H+JPjYK5ZOgqF9cqJZp1N4/DT2ZjPHZrJnjq+yxzUOHPXUWX8dHQ1vPBTrPuF8hmT8rDs4dOIl8uZMiATx42LElJvo2ioj3nsF/8ZqV0+hBD6Ffa2qydW6eA62yJ4lAapXTIod+58jJgn8ZD2Pm8DQkk2hF+gd/vfn4YKVeHP2IpJUKguHlMlwZ/wMJKlcDokrlta1CfD2gVOmDDhWrjYCPgT/FvMtmIScYwfhRNXGJu/jjbiB8IgTLDCubt2KKlWrwt7BXvcb1zcqeHl9CPM71zClj6Xb8No/dNhw9T8n5B3at8OG9evQtNlP+GflCiXKPJ56KItbYFDw94hCkvH5jvFCrjclSpZU1zhi7DoX0fU1sj5RHdfd/SEGDvwDpUuXQfPmLcK8z7j/rt2CQ8RoIOnapZMSxtUicaFbFBGIX69ADO9GbqyUC2da+m3CgxY/CskuXbqoMjGckRFjSReczRibrRnbr+iUl2H83/jx45VLXBOr69evV/GI5mLK8Tg5OYU5FloU9dvQqsg4S7quTWX06NEYOnSo2gfNCpcnT140bNRIl31Od7o+Tz2eqkSd5MlTGB3TlD6WamMMO3t75ZIPLRzrNwgRzvOWTkAcW2fYxo3cepgxnRMSJQyOMXr2ItjF6P7YG5XKJFVx01ooDi2C6r1Pws4YtDJu2BZsvdTEJ+MLGZtIcmdPhIePvAzGYJwiXaE5syYUgRhNmFySJUOIK/TY6VfKGqg+t2TxcOtuyLUoZTIH3LgTcbbxibOv1UIYMpA1oxOWrguOS3z0xBtHT7802B4nDold7A3c0DFFoK8ffB49hWM6w6xZvva6F3EmPK2A+pbAzL93w9sLV3SvWS7n/fVbOnFIPPcfQY5RwYkdpvLSBnD/NL96e+e2MiakSpVa587URBrXM5a9ZKkQkaqPKX0s1SZ16mBPFDN4NWgsoIfm7r3gEJbMWbKoyTatbSQIwSKY4/Carh+rWLBQQRVLzusZt8sYew2+TpkyldFjNqVPVMZ95O6OP/r/ripF9Oj5a6RJjbwus9QNvVSfVSAKZvPFbLAnTpwwEHx0CTOYlen1oeH6nDlzqphCfUJnOVMUsQ0FXZUqVZR5m2hChjMZjYsXLxq1krGcjAZ/oIwjjI41kjF8zETW77t3716DNrTmMZjXWLZ2RG1MOR5ul+VZ9IuH8xzrU6ZMGRVracxCSjFpjP79+6vzSiHO8blMmToNlStXUe8XL1FCZb/xwqJxYP9+VfuLnw/x9PRUs2XOpk3tY6k2LHOjD5NXjh49ogKyNZhVrx+TyPIPPm+OIJ6LaWWG5iy9h/EzbqplyergG+yRU57KelSicEj2Ji2DTEC5euOtrqwNM5DTpg62GlBMJnYLcV3SFdmkThocPPYCL18Hfx+ePPNRQpNlczSyfbJ80eIoRI9jZ15jwpy7uuX5Sz+8efcRF6+9ReUySQxqFmbNGF8JPI2i+Z1Rqoir7nXSxIbu57rfp1A31V0HnxvdFhdaFznmjMXhx39Zkuc79yNZjaqwsYurq2+YtFoFtV4jUf7cSNWkju61Q8pga6eGW7mSqo37olW6dc+27wuOSdQTEQlzZ4f3g6gl4pTwj4Om3rZqoWWKVi16SDjB5TVa47///lPXp+LFiuvWHT58SMWVE1P6WKoNRRfvAZr4I7yOP3z4AClSBNdL5HWTx6MtnTt3DV5fpQp+7fWbwXsUajSWFChQQHk3NHi9u3z5srr+aTBz+siRw+p/U/qYOi7vOf3791NeFe5f6HqOvJ4ynl0fnhP3R+7Kxf45kSSVr9iCSHHQuHFjlTXMNHgmg1B8hM7u0vjrr7+U9Y1WwZo1ayphM3PmTIMfKKEQ2Lp1K3744QdlUdy1a5fqw8znP/74Q2UlU9wMGjTI6Mxn1qxZKg6PNasWLFigttOjR48oHx+zspl0M2nSJPX/zp07MX/+fIM23D7j/Jg1TNFHYcvZpSltIjseFtlmUWy6j0eMGKHO96+/Gs72KlWqpOIPeV6ZNU73Ol38c+bMUVnTzGoODS/MoV0ODg6euv/pcuCFZsAf/VG5SlVV9JXld8b+FRJ7+eiRu6pFmD1HDjWjNqWPpdrw+3Dt6tXgYrU2Njh48IC6aLdu00bXhudq7JhRKFGipLqx7dq5E3aOmeHkVg3R5eEjb/yz0V3VKFy/9TESJYyLWtVSYtjf1xAQGOJy7tMlG0ZOuqba85MaPzgvTv73Eq/f+qNy+WTKDfn3zJu6cbfseoo61VNh+tgC2LnfA/Hi2aJO9ZT499gLXL4e+56Yk7hCCcTPlgGuJQqo72K69sFlgzw27YGvh+GNJTYyfdF9TBySC3/+mlUlGf34XTIcO/NKiTyN2tVSIEF8W1X6htStngIpkzvg8o33ykpYvkRijJx8UyfyYwN3Js5GsaoVUGjVXHgeOIJkP1SG34tXeDB3ua5Nkirlka59czz+J/jhBa4lC6sahy/2HoZDiqTq/1ujp6h6hxr3Zy5Csh+/Q6FVc/Bi7yEkyJEFKer+oMremAvFScdOXTBm9Ej4eHvDxdUVW7dsRo2atVSpFY1//lmpXJ4MgTGlj6XaEAq+wYMH4f2HD8rFevxYcMKjOZ6kNr+0Rd8+vTFixDCVZbx3z251PStbtpyuza7dO+H+0F1dE03tE1kbeqT+6N9P/c9r947t23V9tcLeHz8G4M9BA5AjZy6kTZNGTbYPHNiPVClTxvjDGcIgLmazsQn67JGjQKtWrdRNmeVVKOb4JWrUqJESVJqAGTVqlIo9nDx5sq7fkSNH1GuKGIomikpaFsnAgQOV+5QuUMJYRJbSYS1DJq0wloxtaGljHUGKJbqiJ06ciHLlyikXI10Fq1evxo4dO9TMiTNAxhGaakGkaX7evHmqrAxZu3atEpwUcLw40apJsabv0mUBVa5jhjGFKeP6Qo9jrA1dqREdD+FMjm1Onz6tLLMsG0RBTte31obnjOeH8ZuMeWH8HcvzGBOH4XHrtmHGL8dkCZk7t2+rc8ryD8mSJzdwUbBeYZOmTXXlHyLrY8k2165dxX9nzigrKbP0ypYrpyvtoD973rd3r7oo5smbFzNWWGb2W7KIGwrnc4GffxAOHH2OG7ffGzyho3vbzNi+1wOXrwdbFSk0fqicAsmSxMPNO++w99BzfPxo+JOly7JyuWRInza+epIKC28fORki2k2l/47gGMyYJM3P9eBSzDDGmNz5ex687oRfbNoSjKu9yCLj0KVcpVxS9SSV67ffY++RF7qwAcI4Qgf7OFizJSRzvnRRV+TLmUgJ/X2HX8DjhXELvQYzms9ffYuTn9zSpjLgUHDiX3Swc3VGqsZ1EC9NSuUaphBk0om+QHQrUww3BodMuJyLFEDS6hUQ6OuP59v3qjI3xopwp2pUG05ZMsDP4zk8Nu+G9/3of9YZj4Yk2hHeDw4f+lfFRufJm09NyPWhQKS1T/NymNLHkm1YQeHA/n3qHsMEyIqVvtOFTYWGbmrWQaxStZq6FocHXdD79u5R1jk+8aRipUoGpWWYFf3q1UtVV9HUPpG14f5rNWJDo7+/tCKy+gbPjZOjoxKbrA2pb5zIkjlERMcUXv9apgi/U7ng8ClT4b2Z+uXChQsqHrVz584RPuSC96xVq1Yp7yd1CjUNjVL6T7GhBujWrVuYvuxnzOv61QvEL/1YvNBoApHuVk2YCdETiNZIq56x76kkluZzCMQviaUEYmzGHIH4tRBaIApfH59FIB6Kfj1ffZzKhk3ijEhH0ItFcUeDDEPpaLSi0em7774z2kd7KAi9nvQSLl68GJs2bVLJtpoBjDkNNDCFDgfjtsKbbFj1o/ZiG4ybC6+OHz8gPnpIEARBEIRYwBd4ksrcuXOVFZbWQ4a7/fjjj7hx44by5IUnEKdOnWpQLonhYfSQcqwJEyYYtP3cxqsvIhBpZo+s3tPnhskgLGmjJTOEhvUD6Zo1BmMCWVNQEARBEIRvk7179yohqK8j+CCKn376SYXSGXt0b+hamnTFM9fAWKIo8y8YzkCtwrCy0IXNrUIgMp4utuHo6BhhuRe6xbkIgiAIghC7sdRj8nw/1f6NLFmT8GlkWny/fpkfRvIx4TV0neXwnuZ29uzZMI8Srlu3rsqrYEwoE14pEhmbGJ0HWpiKlBoXBEEQBMG6YBazBZbRo0erSij6i5YMGxrGEoYWjjQ+ae9FBpNjaW1k/CJjDjWYNMo6yqw6wscH02PJ+ES6rmMSiUEUBEEQBMGqCLJQmZv+/furR/DqE95Ta9zc3FTd3ND5C4QVVSKrnUz3NC2FLI8X0fZYZoll6tatW4eYRASiIAiCIAiCERzCcScbo1ChQgYP2yDMZGZ2MutEhgefMsPaxt9//71KTons6TRaOZ3oPOUtKoiLWRAEQRAE64IiyxJLFGjTpg2uXr2qaiBrT55hTWOu10Tf8ePHVTYy4xUJn0amiUPGFoZ+Og1hrWLGMGqw7M3KlSvRpElIncuYQCyIgiAIgiBYFZZyMUcFWhD5hLeff/5Zuab5qFvWNxwyZIiuDR+cQasiH8JA+IALWgOZJFuqVCldOxZd54MvCB+QwfI3rP7ChBe25/h9+/ZFTCICURAEQRAEwQJ06NBBJZrQMsgnqYR2LVP4seA1n4BGGG/Ix7uGxsXFRfc/RSYXPpaY5W/4jO/wHktsSUQgCoIgCIJgXViozE10YG3D8B7Ryyxo/YLXppS+0dBE5edCBKIgCIIgCNbFF3AxWxtyBgVBEARBEAQDxIIoCIIgCIJVYaknqXzLiEAUBEEQBMG6EBez2YiLWRAEQRAEQTBALIiCIAiCIFgVQRAXs7mIQBQEQRAEwar4EoWyrQ0RiIIgCIIgWBciEM1GJLYgCIIgCIJggFgQBUEQBEGwKqTMjfmIQBQEQRAEwaqQGETzEYEoCCYw5GxXqz9PI2ovgjXTZ1MrWDtDq8+BtWPd31JBiD2IQBQEQRAEwbqQJ6mYjQhEQRAEQRCsCnExm49kMQuCIAiCIAgGiAVREARBEASrQp6kYj4iEAVBEARBsCrExWw+4mIWBEEQBEEQDBALoiAIgiAI1oVkMZuNCERBEARBEKyKIHGQmo0IREEQBEEQrAp51J75SAyiIAiCIAiCYIBYEAVBEARBsCoki9l8RCAKgiAIgmBVSB1E8xEXsyAIgiAIgmCAWBAFQRAEQbAqxMVsPiIQBUEQBEGwKiSL2XzExSwIgiAIgiAYIBZEQRAEQRCsCklSMR8RiIIgCIIgWBUSgxgLXMyHDh3C0aNHEdu4ceMGVq1ahTVr1ujWHT58GEuXLsWxY8ci7b9o0SI8efIkhvfy8297+fLlcHd3j5GxBUEQBEGwDsy2IM6ePRvx4sVDqVKlEFugKPzll1/www8/IFWqVGjYsCFat26NgwcPokyZMggMDETJkiUjHIPtd+/ejZQpU362/Ta27aCgICxevBjVq1dHihQpzB67Q4cO+Oeff5AmTRrEFB8/fsTevXtw5/ZtODs7o2Kl7yI9j6b0sVSbV69eYf/+ffDw8ECSJElQuXIVuLq6htmnR+7uOHToX7x9+xYpbQORNyB68yn75EmRqmFtOKRIhg837+Dxqg0I9PGNsE+ignmRpGIZ2Lm64O35y3i6YSuCAgLCtHMrWxJu5UogyO8jnm7cqsb/ksRziIPvKyVDutSOeP7CF9v3P8erN/4R9kmV3AEVSiVGUjcHuD/xxrZ9z+DtE2i0rZuLHdo0SQuP575Yuu4RYiU2NkhatSyS16yEtxeu4cGcf/C1UaygK4oXdkNAQBAOn3iBC1feRtjeLq4NqlZMjkzp48PXNwB7Dz3H7XsfwrT5rlwyZMmYAIGBQbh+6x0OHHmOAOMftcncvn0Lh/79Fz6+PsibNx9Kly5jkT6WavP48WPs3r0Lr1+9QucuXWBnZx+tNrwXcFtXrl6Go6MTypevgAwZMkR4nKb0MbXNuXNnlTEoderUqFOnrsH7J0+ewLFwDEXtO3SEo6MjPjfiYjYfq0xSoaBq3769EkITJkxQomHZsmVYsGABlixZgp9//hmxGe4fhS0JCAhQgvHatWv4GqD4/vPPgVi7Zg0SJ0mCBw8eoGuXTrh166ZZfSzV5s6dO2jX9hdcuXIZqVOlwt07d1SbJ08eG+zTvn170bVrZzx7/lx9FifiBmKbXViBFhmO6dOgxK61cClaED6PniB1s/oosn4J4jiEvQFopGvfEoVXzYOtkxO8HzxC2l9+QoElM5Tw0GFjg9xTxyD3lNEI9PaB/6vXyDVxBBLlz40vKQ6nj8qD6hWS4ukzXxTM64x54/MhWZLwj7VYQRcsnFAAaVI6KnFYtIALZo/Nh4QJws5defgDemRFiUKuKFPMDbERhxRJUeHKTmTs3hJupQsjSaWIJ6KxkVZN0mP477ng7R38fZ80Ij9qVw9/gucYLw4WTC6Mej+mwrPnvrCzi4MZYwugbInEBuJw5l8F0bxBWrzw9MXbd/7o3DoTxg3Ja9a+Hj16BL/27AFvb28kSuSM6dOmYvr0aWb3sVSbqVMn489BA+D+8CF27dqJjx/DXkNMaUMm/D0e8+bNgYuLK968eYMe3bvizJnTER6rKX0ia/Pu3Tt0aN8Wa1avxvVrV41uM3HiJMiRI4fBcv36dVy8eEEZkL6Ui9kSy7dMlC2Ir1+/xr59+5AwYUIULVo0zPu7du1C4sSJlYXqyJEjiB8/PqpVq6beo9uU6+LEiYOyZcsiadKkun6aiPvxxx/h6emJS5cuKctP6dKlw2wjonEoACmmHBwclKuWvH//Xo3/77//4t69e2p/omMZNHX/eY4uXryIZMmSqf230b+xA2r/zp8/j/Tp06NIkSLKFV6+fHmdVa9ChQo6ixZFLtmxY4fad1rFqlSpgrVr16J+/frqc9Cg+7xixYoG1kG62s+ePavbljFoIeO58fPzQ4ECBZApUyZEl8OHDuHSxYuYN3+hOn4yePAgzJ83D6PHjI12H0u1WbpkMbJly4aBA/9Ur2vVBkaMGIZFCxei/x8D1LqnT59iyuRJaNu2HWrUrKXW5Rk/D69tgqJ8PjL36Qavew9xrlVXTsPxeOV6lD62A6l/aoCHC1aEaR8nngOy/N4TN4ePx8OFwe8/WrkOZY7vQvLa38Nj4za1LnXzhkhWvRKOV20A77sP1LqHi1cirt734XNT9/sUSOxij2Zdz8LLOwBrtzzG7L/yoVWjtPhrxm2jfTo2T489h19g3Mzg99dte4rZY/OiWZ1UmL0s+Lg0mtdLrUTLoRMvkStbAsRGAj544cT3beB9zx2F107H10bSxPZo1TgdRk66jt0Hn6l1r177KTG3c78HfHzDmvtqV0+FJG72qN/mhPrcCS28v3XMiiMnPREYCOTKngg5sibET51O4b67l2pz5cY7TBmZXzc5iCqcEM6aOVNZs9r80laty5kjJwYO/APff/+D0euYKX0s1YbUqFETXbt2x/Fjx5SgNIYpba5cvqwmrRMmTkb27NnVOt5WZs+aiTlz50e7jylt7OzsMHjIMGU5HDdurLpfhCZz5sxq0fjw4QNmz56FJk2bhbn/fS7Egmg+UZLHFy5cQNasWTFs2DBMmTJFiQmu0+evv/5C586dlQt3xYoVSpwQWu/4o6FLeurUqcqErR8f6OPjoyxlTZo0Qc2aNVWsHP/SPUzztkZk49CNzFnQ3bt3ceDAAbVQ1JHTp0+r13QxRhVT9/+nn35C3bp1sXLlSvW3QYMGBuPQopk/f34lXn/77TclBukOP3funK4Nx7l8+bL6X9t3vs99P3nyJF68eKHa0EWqT7t27QzGmTZtGvLmzWuwLQpZfbZt24aMGTNi4sSJ6pxTRA4cOBDR5cTJ48iZM5dOoJEKFSri0qWL8PLyinYfS7Xx8HiqjlefTBkz4dSpk8paS/bs2a1mvd//8KNBO5egqF/oknxXDs8271TikPi/fgPPg0eRpHIFo+3pho5jb4d3V67r1gW8/wAf90dIWrWibl2aFo3gsWWXThySIP+P8H8Z9e+2pShV2BXHz77SiQS6Dg8c9VTrwyNlMgfcvm/oirxz3wulixpaCPNkT4iaVZLrhGRs5eO7D0ocfq0ot3IgcPDoc906CsX4TnFRMK+L0T4pk8fDYw8f3edObt19jySJHZAzayL1+sVLX+WudnG207VxdbaDt08A3ryLOAQhPG7fugVPzxcoXyHkt1SgYEG4uLgol2d0+1iqDcmYMVOkAsmUNidOnkCKFCl1Ik67tj169Aju7g+j3ceUNrwWUhxGhQMH9qt7DcN3vkUCAgJUrD+FsiX7RGfczyYQu3XrhkqVKinRt3nzZmWto6UsNLRaMSGEVq7ff/9dxVewL0UlY+v279+PQYMGoWPHjmHEGm/ktB5u2LABp06dwtatW9U4xJRx5s+fr2Yy9erVU8KICwUdobDl61y5ckXpJEVl/3mBoGimeOQ5WL9+Pc6cOaPe4wf7xx9/YOHChdi+fbsSf4zd9PUNPx5N23eeR+776NGjTd7nPn36GGyrWLFiBtt69uwZGjdurCyfe/fuxbp169S+Tp48We17dHjy+AmSJ09usC558hRqxk3LXHT7WKpNxkyZVCyNv7+f7gf333//qfNCy7V28c+WLbtyR8+bOweLFi7AZduoB0rZJ02MuAkTwNvdMFbO+6E7nDKkNdrH5+Fj+L95q4SlhmO6NHDKkhFOGdOp1zZx4yJB9ix4c+Y8ktf5AdmH90eGbu3glDE9viSpU8RTrmV9njzzVaIgvpOt0T637n1A8YIuOu853dT5cydSY2kkiG+LgT2yKHH45p3hBEewLLTm0WLo5x8yKX/xkq8D1XvGuHn3PTKkcUKq5CGfWckiwe7ltKmC+zx64oMh466gd+esGN4vF0b9kRstGqZD/xGX8O599D7Tx5/CQpIlC/nNU2hxgvjk8eNo97FUG0vCMY1d29S+hLM9U/pEZ1xT2LVzJ4oVLw43ty8XCvKlXMxr1qxRHsp8+fKp46dOCG2YiU6f6IxrLiYf/fPnz5UbsmfPnrrZTrly5Yy6LWvVqqWLodOsVAxSpaVMg+PQ9Ut3tT60PtI9TCj0ateurYRLVMexJFHZbps2bZQLmnBWxiQICmZtHArIpk2b6tr/+uuvMbLPFNZ0R+tvi1ZEfShe48aNq4QR3dNcmJXOz46W2Ojg5+cbJiDZ6dNrvhfdPpZq06pVG/j7+6NTxw74a+wYdO7cEU5OTgZtvL298PDhA0ya+Ddc3dxga2uLhQ4fsco+aj/GOJ9ibwLeG1pO+Vp7LzRMRLnaZzDStGyMwmsXIc+McSj0z1y8u3QNtp/62Do5wsbWFuk7/IzkNarB6+59JQ5L7FmPxBUjD9CPKezt4yiLkD6aVcnB3vilZvKCu8iULr6KVRzYIyvm/50fj576wNbWRsWtkb6dM+PwqVc4df7NZziKbxt+TvqWQA269h0cjH+G2/c8xalzrzB/UiEM7ZsT00bnR5YM8ZXFUOtjGwcoUzyJmghcufEWV2++g5OTLUoVDYlTjCp+vsGTPO33q8FEC4bLRLePpdpYEl8j1zbtdXjbM6VPdMaNDE6sGfddvdr3+JLQxWyJJSpcuXIFzZo1U8aoly9fqlAy6peIDDum9InOuJ81BpEB/yRdumArhgZj20ITOtuWfdOmTasTTprZmmr4/v37Bm2NjU9hGtVxLElUtksBqA/Frma1e/jwoRpH353AMSjSYmqf9bdFN4H+ttiG4mfPnj0GfRkeENoNq8Fj4UKX+vjx49W69x8+IG+efKhStaq6YFI46/Pu/TujF1MNU/pYqg3jRmfMnK1c+HQR1atfX1l2GXjNQHPi6OSkRPPfEybCzS34BhZ/0RrMj/cRVf2D4GqiqzngXfC+xHUOdrNp2Lk442Oo/dTn2bY9eHX8DJwL5YNtfCfcGDIW2Yb002UxB3h5q//9nnviQtseun42cW1VzKPn/uhZf6MUp9YwJM71yOlXOHo62LWcIL7hdznRp2STD17GA+9v3fVCi+5nkSdHQrgmssPiNQ9RtrgbcmdLCP+PQciY1hHliifG7n+fo0/H4LiuXNmD2/L1P/97jIePfWL0eK0VJqMkTxo8GX/z1h+zFt9Vn5OxBKH48eOG+xnSJf37iMvIljkB0qdxUlnr9x9+wIZFJfH6bbD7uHL5ZKhYJikatDmOl6+D1x06/gLLZhRVcYpnLryOcF993hyBv3dweMHkSfHRsVNn3W/6/ft3cHZ2MUiqSBFOjLkpfSzVxpJwey89Xxqs47b19yU6faIzbmTs2rVDXWcLFS6Mb4158+ap8DNa9wiTddq2bYtZs2Ypr2N0+0RnXEtgsjLREjKoXvXjEfg6tCgKHU/Bvpr7Th/21Y8X09aFfq1tOyrjWBJLbZfWxNDHxwtKVM3EFHWErlMNjqE/Dvc59LYYXKzfhhZGoiXzmAJnLEOHDlX/J0oULHy+++47VK0SnIiUPn2GMBnXFNH29vbhJgaZ0sdSbQhFMuNANbZu2YK0adPpjidDhoy4dfOmThyS1IHB3+mXNqYLRMYb+no8R/xsmYCtIevjZ8uMD9dvRdz35Su82BNsxaW10LVEEVUehwR9/IgPt+7i3dVgy7TG+2s3kbh8zJebYqLClZshAve5Z7Cl4e5DL+Vq1CdDWkcVn+brFxjheKf1rIOF8zrjwtXgQPgXr/zDxB0y5s3RIY7ah/BEixA5d+69V7GBxOvTebxz/wNcXeyRKGFcvP3kzs+Q1glxbW3UexFx4/Z7tZCqFZLhY0AQLn36HFOnCHZda+KQ3HvoBX+6rlM5RioQ49glRVwEf4dy5HBTk/X0GdLrJrp5P8VHMnSEyYRM5DOGKX0s1caS8Np24fx5dc3XDBWacSJduvTR7hOdcSOC5+DA/v2oVau2gUHlW3kW86lTp8KU0GNpvTFjxiiXvb5nNSp9ojOuJTD5E2RmLK1KmruXMJDVlCLZzPilleb48eMGLlCWBwhdP1F/fL6/ZcsW5cqO6jiWxFLb5ThMnqF5WN/NGxEUM7RWclv6FlqKRJYR0GBcpJZkob8tLUmIrF692mDs77//XglfLVNag1Y4JsIYo3///ioJiAstolyGjxiFPHmDy1WwhtadO7dx+fIlnZti544dKFWqtK62F4OfJ0+aqMIWTO1jqTbcpn5szc0bN1RNxIaNGunWMZmHmeg3b4aUx7lkGwi7ICDlJ6FoKk83bUfK+jVhmzA46zZ+9ixwLVUUTzdu17VJXrMasvTvqXudqFA+lc2skaFrW9jGc4D7klUh467fArcyxUPaxYmDxBVK4+2FK4hpGDO2de8z3XLzbrBw2Hv4BYoWcNbFD9J6WLlsErVeI3f2BMryR3c0SZ/GUSUraJQq4ooCuZ2xcuNjo9vict/dW8Ui8n99wSFEjX+Pe2LLrqdq2Xc4+Ld44r+X8Pb+iLrfh9xwGtRMjafPfHDpaoiI79ctG4rkDzEMFMoX8j/FZesm6bFz31N4vvILSVpxc1BWRo0Shd1USZxbn74/EWHvlA2OzmXVUq369yqzNk2atCoMafPm/+na7d61Gx8/+qOUXvWLFSuWY/euXep/U/pYqo0lKVumrJrgsy4roaDbumUz8uTJqwwP5M2b1+q6evfuHZP7mNImKjDWnQkUVaoGGwy+JEFBNhZZfH191TnSX8LLG+B9k1Vc9NHOY3j3VFP6RGfcz2pB5GyAGcqMaWNiBl2/c+fO1VldIoLZzizQzNhExu7RivX333+rJIrQpQjo7mzZsiUKFy6sMoFZJqdTp05RHseSWGq7hQoVUkkhLIXTvXt3ZT1kzUaKwIiy2JhcQncuBTm/JMyObt68uUqcuX37thIzLJWjP2MrWLCgyqjmtnr06KG+1Ewq0ncxM9iV1kCeb8YeMnmH8SP/+9//VGKQsQsEXeZajGjIuhDrKoVi/QYNMfjPQShQoCAePLgPmzhx8Evb4FIQhJZN1vtiCRlaOk3pY6k2PEdjxoxCYloHbaBmz40aNcZ331XWteGsmqUr/ujfDwULFlKJU1fsA/CTry2cohiTcmfiTLgUK4gSO9fg3aWrcC1VDE83bMOzrcE3LOJcOD+SVquEW6Mnqdf2bq4ovn013l2+Bsd0qREvbRqcb9sTvk9Cstbvz10Cl+KFUXLfRrz57wIS5MwGG7u4ONci+LfyJdj97wsUL+iKGaPy4NyVt8iROYFKWlmxISRJJ21KR/xYOTlmLrkPhjjFsQEmDM6FB4+94RjPVmUrT5h7B+cjKcwcm8k9eZAS7ony52AgFPLOHqGym6/2jtl4IUtAi+zoKTcwqFcOFMjjooQ8i1//MfKSQUHrmlVTKovi6fOvdSKyTdP0eOHph8IFXHDx6ltMmH3LQIzuOuCB6WMKKBFKi2TRgm5Ysf4hLl+P/mfdo2cvVUOwZ49ucHZxwcULF9ChYyckSRJSgoylY3idZgiMqX0s1YbWtPPnz6mEQDJz5nTYxrFFrdq1VfayqW1Sp0mDX9q2UwKQ7SkKOEkfOSrkO/Xhg5e6rpYsWUr1M6WPKW0Ik/Uo/K5fu6ZiuNme95IuXbsZtOP26VrWLwH3tTNaz2umMXjwYAwZMiRMW95feH700WI5Nc9fdPpEZ1xLYBOkX0PGBBgPyGwa1t+jBerq1atqNseyK2TcuHHKV87yNPpwM7SW0dLFg61atSpq1KhhYLXimCzlwoBMxojRUkZxqK+cIxuHDB8+XIk6lsnRxu7atasK8Awd4xgerVq1Qt++fXUZz5FtlzMKikjGA+jXg+rVq5fKqKY5mFBcsmQOM2cpslkGh8ksTAqh1c/YtpmBSzFOdwZFG7+wHIdxCcwi5/lmce0BAwao49Tcp2xD9zHL+3BbLVq0wIgRI9Q51XexnjhxAps2bVLCn/vCYNiouM5v3b4bZh1rNtKalyhhIuQvkN/gyQAvX3ri1MmTKFW6jEEdx4j6WLINf1gUhrTK5syVK9yZMs/7tatX4ejkCKe+I5EwiuJQR5w4cC1ZVBVR5pNO3oWy8tFi6Jg6JTxYDkcvA9qleBEEeHnh1dFTCPQxHmdHcemYPi18n3rg9cmzyv0cXUaUmANLkCNLAqRNFU+5nukq1ouEUNbFArkTYefB5/j4MUhXaLlQXmfEiWOjntgRWaYyayDS6njkVNRK+vTZ1Aqfg9Qt6iCOneHcO9DHD49WhFibYorR1S3zGSZ2tUeBPM4q0eS/i6917maNGlVT4PK1t7j7ICQBK3f2REiVIp4SjqGfoqLBGMVM6Z2U2Lx1570KP4gqiyYZXsM5gTt/7py6BvOamSxUVi4FIq8FmpfDlD6WanP16hU8MBIfX6hwEZ2IMqWNBgv6M4yGiSScBOsXofby+qCeiBK6X0R9TG1D8ejra/hZxbG1RZUqwaJbu0fyaTC8h3CSHRFZMhuPcbckN29bJi8hXZoUYSyGxgwlhE88S5Agga7yCqFeatSokbq/hg7HM7VPdMb9IgIxptAEIp+TXKJECVgrdHHq/3j5gdPSx5qGMfUhxzTGBKK1ca98cMFsa8ZSAjG28rkE4pfEUgIxNhNaIApfH59DIN64bVhoP7pky2z6940xgTSSMRaVse+ExjPWJ9bCvTipYJgTjTY0rpnSx5Q2MYHl02e/AmidZNCnMfgEE5bWiSlYS5GxgRTBDAaeMWOGqo34tYpDQRAEQRCgvIi8x9NbR+8hvXMsH6dv+aMXVvO+MhvZlD6mtLFqgUglTTfp54hdYGIFXdnhJePEpEBkLAM/VCb30GTMZBcmRQiCIAiC8PU+as/V1VXF8zPUjDWRGarF5NA6dero2jCvgmFomiXQlD6mtLFqF7Pw9SIuZutAXMxfP+JiFr4GPoeL+dptyzzyMkfmkJqv3xqxxoIoCIIgCILwtVoQrY0vW8lSEARBEARBiHWIBVEQBEEQBKuCRa4F8xCBKAiCIAiCVSEuZvMRF7MgCIIgCIJggFgQBUEQBEGwKsSCaD4iEAVBEARBsCpEIJqPuJgFQRAEQRAEA8SCKAiCIAiCVSFZzOYjAlEQBEEQBKsiUAplm424mAVBEARBEAQDxIIoCIIgCIJVIUkq5iMCURAEQRAEq0JiEM1HBKIgCIIgCFaFWBDNR2IQBUEQBEEQBAPEgigIgiAIglUhLmbzEYEoCIIgCIJVIS5m8xEXsyAIgiAIgmCAWBAFwQRGlJhj9edpwKF2sGaGVrf+z7D/jvawfnZ86R0QvgLExWw+IhAFQRAEQbAqAr/0DlgB4mIWBEEQBEEQDBALoiAIgiAIVoW4mM1HBKIgCIIgCFaFZDGbj7iYBUEQBEEQBAPEgigIgiAIglUhLmbzEYEoCIIgCIJVIS5m8xGBKAiCIAiCVREY9KX34OtHYhAFQRAEQRAEA8SCKAiCIAiCVSEuZvMRgSgIgiAIglUhSSrmIy5mQRAEQRAEwQCxIAqCIAiCYFUESZKK2YhAFARBEATBqgiEzZfeha8ecTELgiAIgiAIBogFURAEQRAEq0KSVMxHBKIgCIIgCFbF1xSDuGvXLly4cAHJkiVDnTp1kChRogjbe3h4YO/evXj+/Dly5syJKlWqwMYmxKV+7949LFu2LEy/Tp06IXHixCbvl7iYBUEQBEEQvgBNmjTBzz//jNu3b2P69OnIkycPHjx4EG77CRMmoGTJktiyZQvu3LmD9u3bq9fv37/Xtbl16xYGDRoEHx8fgyUoiqr5i1gQlyxZgrhx46JZs2aILfDErV69GqdPn0a8ePEwfPhwvHnzBgsXLsTdu3dRqlQpNG7cOMIxunbtim7duiF79uxm7YulxjHGb7/9hjZt2iB37twWH1sQBEEQYgNfQ6HsDRs2YO3atbh06RJy5MgBf39/pTX69u2Lf/75x2ifokWL4sqVK0qnkCFDhiBbtmyYPHkyBgwYYNB2xIgRZu1f3C9lTuXBxSaBOH78eEydOhXdu3dHkiRJ1LpatWrBzs5O/U2ePHmkY1D90zxsqrALDAxU2+vRoweyZs1qdJzw2kSX2bNno2LFijEiEA8d+hd79+yBl9cH5MiZC40bN0H8+PHN7mOJNjOmT8P169cN+uTNlw9t27Yzul///LMSx44eRYMGDVG2XDlYAnv7OGhSKxUK5XWGn38g9h99ge37nkfYJ3WKeGhUMyXSp3HEe68A7Nj/HIdPvjTa1i6uDYb2zobErvboN/IqXr/9iM9F6hYNkbxGFdjY2cHz4FHcn7kIQX7+4ba3c3VG+i6tkSh/bvXQ1Gc79sF90aowfiHH9GmQrkNLJMyVDd4PHuHulHnwunUXX5JSRd1Qs2pKJEpoh2u33mHJ6vt4E8G5trezQf0aqVEwrwscHW1x9uJrrFj3ED6+gQZt6vyQCoXzuyK+oy0ee/hgw9bHuHrzHWIjiQrkRPoOTZGoYG7cGDwZz3f+i9gC3W6r/lmJ+/fvwdXVDTVr1ULevPnM7mNKGw+Pp9ixfTvOnTuLChUqoXadOmG2df7cOezatRPPnz9D/AQJUKxoMVStVh22trYmH+PZ//7D1m1b8Ob1a2TKlBmNmzSFm5ub2X0ia/Py5Uts3boF169dU0aevHnz4scaNXViRePChfPYvn0bHj96jF97/YYMGTLgc/M1PIt57dq1KF26tBKHhHqjVatW6N27txKLfB2asmXLGrx2dXVV/WlNNHa/9/X1Vff7SpUqGbihTUFczJ/YvXu3MvPyg+EHRHPsoUOHMHLkSCXQKlSoAEtD8Ucx+OjRI4P1FKraFya8NrGNnTt34O/x41C4SBE0aNgIF86fw6CBA9T+m9PHUm3cH7kjfYb06Nq1m2758ccfje4XL5L79+/Dgwf38frNa4udoyG9suK7Momxdstj7DvyAl1bZUDzeqnDbZ8mZTzMG5cPcePGwaLV7koY9u6QCT9USma0feefMyBZYgdkz5wAdnE/30878+/dkHVATzxZuwUP5i5D6mb1kHf6mHDb2yaIj2LbVypxeG/qAjxctApp2zRD9pG/G7Tj+8X3rIGtYzzc/ms6nu/cjzxTR+JLUqFUEoz6IzcuXHmDpWseIFvmBJg6qgDixg3/wjt6YB78WCUFtux6gsWrHiBfLmeMG5LXoE2/btnRqFYa7NzngbnL7sHLKwDTxxZQ48c20ndqhnxzR+HN2StwKZwH9oldEFugm633b7/i5auXaNrsJ6RNmxYD/uiPixcvmNXHlDa0AvXv/zscHBxU++cvwk7+/vvvDAYO/AOpUqXCz61ao3TpMli8eBEWLlxg8jGeOnUSgwcPQpYsWdGkaTN1b+jb5zd1zzKnT2RtKDR69+4Fu7hxUa9+fVSpWlXdN4cMHoSAgADdOHPnzMbyZcuUwLx16yZ8fLzxpZJULLH4+vri7du3BgvXWYJr164p658+fM1zfv/+fZPGYLzhqVOnUKJECYP1SZMmxYkTJ9T3smnTpkpYvnv3LvZZEP/77z+llBMmTIiqVauGeX/atGlIly4dXFxcsGPHDqRIkUKJMs3auG/fPsSJEwfVqlVD+fLldf38/PzQq1cv9OzZE8ePH1cnImXKlPjll1+QIIHhhTWicWidY4AoP/hXr16pdZq/fuLEicqiGB2X7x9//KHG5DZ5QalZs6ZO+JHffw++IdI0zPPD4x44cKD60nz48CHcNp07d8aff/6pTMuatZP8+uuvaNu2rc46qO82T58+PRo1amR0P8+cOYPNmzer81moUCHUr18/SjMNCrGlSxajfoOGqFmzllqXIX0GtGnTCsePH0OpUqWj1cdSbTQ4688a6scYmtevX2Py5IkYMGAQfv+9LyxF7uwJUKqIG9r1uYCbd4M/WydHW7Rrlg5rtz4xsCZp1KqaHM9f+mHczNvBKy4D8RzioG2ztNh54BkC9LqULuqKAnkSYc6yBxjdP+Q7FtPEdU6EDJ1b42q/4Xiy5n9qnf+r1yi6aTESTp6Ld5euhemT7PvvEC9NSpyo0ggf3wXHzfi9eImimxbh/szF8Hn4WK3LNXEYXh48hiu//qnr+3z3QXxJ2rfIiA3bH2PlBnf1+sqNt9i0uCSqlk+GbXs9wrTPnCE+ihdyQ8c+Z3Hp2lu17uqNt9i6vBRKF0uMIyc91boyxRMrYbjvcLCoOH/5DSqUTqL63rgdElsUG3Bfugn3Z65Q/+edPhSxCVq3/Pz80b//AGV9KVSoMO4/uI9lS5di7F/jot3HlDZZsmTBvHkL1PX+8JHDRrd18uQJZMyYET81b6Fe586dB+4PH+LIkSPhejNCw+tdpUrfoUmTpup1zpy50PynpsoqWatW7Wj3iawNj3vWrDmwt7c3ECG/9uyhrFeah4vH5uTkhMePH2NRFIRvbGX06NEYOtTwez548GB1/w0NY//Ccw1r0JJHNzLhRMLZ2dngfeog7b3IoE5o2LAh8ufPrwxbGtQZN2/e1I1NQ1e+fPmUbqCmMZUYNzNs27YNxYsXV1k3FCCM49u/f79Bm40bNyoBxvg4Zu9QTBG+pqihWKG59fvvv1exgRocj9a17777DuvWrVMCdMGCBSpgU1/hRzYOhZ+jo6P6svPEctG+7Pwx83VkrlJjcAz2zZw5s/qwGDuwc+dOg/cJxTHbcVuEx/Tw4cNw21B0sg3FjD4zZ85U8ZIaFJJdunRR5n9uv1ixYmFmPnSt83x4e3ur88fzUqNGjSgFsz64f18J66JFi+nWJUueXLkV6E6Jbh9LtdE4fuwo+vTuhZEjhmPHju1hrJs85r/H/4UffvgxUiEZVehW9nzlpxOH5OjpV3CMZ4vc2RIa7ZMoQVy8fG3opvV85Q83F3slPDSSJrZHr/aZMHLyTfj5hW+xjQlcihdEHAd7vNgT4mJ8ffwM/F+/gVtZwxmthp1rIgR4++jEIfF79hw2tra6PglyZUPC3NnhvmS1Qd+I3NYxTbIkDkiXxgnHToW4+N+++4hL198q13B4nyHhZ6/xwStATQiKFgjpc/XGO+TNmQhxPl2RM6ZzUi7sK9eDRWVsIuB9yHc4tsHffIGCBQxcc8WLlcCVK5fV/SK6fUxpw+ssxWFEZMuWHU89PJS7mvj7++Ha9WsGhoOIoAWIyQz61zuKMYbLhHetNaWPKW14bPrikNBaSgICPhr0iw3wFmaJpX///iofQX/hOmPQkho6MST0om9tpa7g/Vwfjq+9FxG8Z9euXVv9ZcKK/nczTZo0BsKT2oZCMrT2+qIWRN5waeGjQBszJtjlRDduaJOq9uXjLEr7Al6+fBmTJk3Cnj17VMwcocBp3rw5WrZsqSxiGhRetK5p1sBMmTIp3zutkKaMQyHFxBkKWSaIkBcvXijrHQVtgQIFonX8rVu3NnjN/eLMgxZMQktnx44dUbdu3XBd2Mba0KQcGbSm8hwcPXpUZ3rmedeOj9y4cUNZOWk91S5QfJ8z4U2bNqk4SFN49vyZ+hs6fd7NLbHuvej0sVQb4uLsgsKFiyjBTfHN2TIvfP1+D/mhr127Br6+fspNbWmSJ3HAi5eGNyjtdbKkhhddjXOX36JXmSTImTUBrt58r2IYa1QOdi8nT+qAG3c+KEExsEdWrN78GLfueaFwXsPZaEwTL3VKBPr7w+95sCVMw/eZp7ISGuPVsTOIG98JKRvXxpNVm9S6tG2CrRaOn/rEz5ZZ/Q344IW8s8fBIVkSfLhzX1kYv1QMIs85Cf05enr66d4Lzc277/Hu/UcVgzhtfrAl+PvvkiNB/LhIkSykz4DRlzG4d05sXFQSb975I4mbPUZMuIYzFywX4vAtwLg+Xmf1cUvspiaDnp6eysMUnT7RGdcYtNB5ffiATh07IFmypKovRVm37j1MPL7num3rk9gtsXLnRrdPdMbVYrVZmiVz5iyw1iepODg46IRwZNDYFJXEELan1VEfGnO4vYjiNik0KQ6fPHmiRB8FYGRQY5lilfxsApExDEwMoP9bgwetmVf1oetZf3Zy4MAB9aPTRB1p0KCBEl0UkvoCUX98upaZVMKTRoEYlXEsDc2/a9asUUKMswSmrjP76HPA4+ax6ccl/PTTTwYCkbMOWk5nzZqlsxjyL2ciDAswVSAGfAyeEYUOqLV3sIefr1+0+1iqDWGgtNYmX778SJ06tYohqluvnprV062/ft06TJ48JVIrQHSIa2sD/4+GVtmPH4MQEBCk3jPGjgPPkTmDEyYPzY0nz3zgkshOrSte0BW2cYL7/NwwLT4GBGHV/57gSxDHLi6C/MMmaAT6+MAmrvHLy7uLV3Gt/yjkGNkfmXp1hG08B7w8fBI+Tzx0feLYB39WuaeMxN2Jc+Dz5ClSNaqNErtX4fh3DeF1x7T4HEuifU7+/oZWWl+/ABUnaoz3HwIw+K8r+L17dlQumxS+foHKmsgYRlu9z711k/TImC4+ps6/jReevihXMgl6tM+C2/c+4L67VwwfmfXw8ePHMNcCB/uwVq6o9onOuMa4cvkylixZrJJXChYoiMdPHmPxooXYsmUz6tWrH2n/gI8fw73effx0LYxOn+iMu3rVPzh+7BhGjhpjNJlCiJx69eqpZF2KRBpmaF2ksYpePO2c0n2/YsUKXQ1DTRxSXzFsjgI9NGfPnkXBggV1r+lloxGtevXqiDUCkW5loh8nR4ypXWbi6PP06dMw7egi5jq+p4+x8Wk5jOo4loTZXkWKFFHilBZDxg6SqAaJmnPuQ58XxjYw80yDs0a6lfnF1IeW0/CspnRRh3ZTx3MMzmCjCNbiJ9TrN2+RNJnxmU3CT4VAI+pjqTYk9AWM2YcUgnTJUyD+e/CAKowwcmTI7I+uo3Vr16rAcqATTGVkv+zKAkQePPbGyMm38ObdR+T65G7UnYMEcZVIoJsyPKYvuo/5Kx8iRVIH5W6mpapxrVQ6K1a18klAj8XsscFJD3RZk9F/5MDOA8+xZkvMCkf/l29g6+SIOPEcEOgT8r2wc3NVsYjh4b54FR6tWAfHdGkQ4OUNP8+XqHQ7WCSqcV8Fu1nuTZ2PJ2s3q/9fHT0N11JFkKppXdwaOSlGj6tWtZRq0Rg6/qr6DAldv0BI4L1zIjtl9QuPk2dfoX6b40iZjC5IG7g/8caiKYVx76GXznXdqHYa9B95WReTePbSG+TMlggtGqVTlkTBkJcvPTFULwaM3pW69eqr68Hbd6Fcdm+Dv0sJExovPmxKn+iMa4xly5ciX/78aNGipXqdJ29eJc44Sf/xxxoGlqp9+/Zi08aNutc9ev6qu969e2t4H+H1LlEi46EqpvSJ6rgbNqzHypUrMHDQn8iVKxdiI19DoewGDRoo8UcDFv9nPgA9hMuXL9e1oYGJNQ35PgUihSKTg2jsmTt3rq4dQ9k0Y9n//vc/tGvXDmXKlFFGH4pDJkYxFjHWCET6wYm7u7uy2GjQxRc6MDM0jENkPx6cljDB+EGKOi1GUYPt9OH4WpuojGNJmPTBWee///6rK1/A+Eh9TEkEMdZGs7TyODQYh6AfY8NzHzrzmcfMfdKgeGW8Q4cOHUyeARoL2O3YsZM6xps3b6hYyeB988Pdu3dUPKgxaEmOrI+l2hiD8Zt0DznGc1SveXOpWLGSQZt+/fqgXLlyyi00YhZMZt7Kh6psCdGSTxh7WP+HFEoU0uVI6DomdBVHBMe45x4sSOr9kEKJkeufEhf6j7mu2xbJkSUBfm2XCXOXP8CdBzFveXp7MdginqhAHhV7SBySJ0W8VMnx7sLVCPvS8uh1OzhcIun3lWAT11YlpZB3F64gKDAQvs9e6HUIgp/nK8QN50ZoSY6c8sT12yE3S48XvggMDIKXdwByZkuIy59iA/nzzJElITbvfhrpzYqla0jK5PGQKX18LF4VbAWN72SrhONLvThF4vnSV31fhLBQlLESgYbLJwNDlsxZcPPGDYO29GLRIBDePceUPtEZ1xgf3n9A6tTB90UNVzc3fPzoryxD+gKxYMFCSJsmrcH1mu9zUn/j5g0ULFRI996NG9dRvLjxmF9amCLrY0obDYrWJYsXYcDAP1XYTmzla3jUno2NDdavX6/0wvnz55WXj7GC+qWFKPxY31Az+FD0aXpKPwtdXw8wlI15F3zaCnUBtQe9tLGqzA1rBzKub8aMGbp1J0+eVEtk0BRKf/nKlSt16zjLojjSdxeTOXPm6IQPxSHj5+hmjuo4loSmYu6TFpBKAad/HgiFDX+UoZNNImtDayQDWJnCrsEZh35iCY+b8S0sxKlfPie0eZv7GDpmgtZXYzWViLGA3a7duqN0mTLYsH6dqkNI+KXnsZfXi62cOnUy5s2bqwsFiKyPpdqo2mQ7tht8FrNmzlDntVDhwmodL/RMTNFflKU5WTJk+JQ8ZCp3H3jh+u0Parn/SdgdOfUSb99/RMsGqXU1C3+qmxr/XXyDJ8+CLW+O8eIoS2DJwsGWUJZOqV0tuUEmdMMaKbF4jbvOXa2/LS7uj4MvGHfue+G5p3H3viX5cP02Xp86h0y/dtC5hzP16Qxfj+d4se+Qrl2BJVNVPUONNK0aw+aTGzle2lTINrg3Hq/apHMdsz/L2qT5uZGunUvxQkiYJ4dORMYkni/9cP3We93C5B+GBOzc74H6NVLBOVHwsdb9nv/bYce+EIHYpU0m9OkSUrO0esXkSJQwrk4M9uuWTdVP/PdYsPh9+NgbL176on7N1Dq3MwUkk1jOXZQYRGNwQqv/W9W8RFWrVVPXLtYw1X77e3bvUnUGNViapkf3bsoKaWofU9qYAusGHjt6RPUn9MZs27pVTW5DC0161fSPkeFA9HqwvMz2bVt1+79nz27lMapSJaRCyNKlS1TpL2JKH1PH/d//NmHRogVKHNJDJpgP7zPUK7QS0lgTujYl4+Z5j9YEIvMS+Dr0wpwKffgIPloZmQdCL2ZUxSGJ8enplClT1HMCqXr5Izh8+LBJ5WLYdty4ceqpHzSPUsiwVA1NqqFdp1TIhQsXVmncNL2y8KRWhDsq41gSii9+aEygYekY1lSkIAkNYw1Ye5HlfWj1Y5kbU9pwYXINk28oHunS1ncfM/6Qlj6anFlehy5YJt7ox3lyLJq3mTjEsekq4EWQbSmyTA3YdXDwROfOXTFq5Ag0/6mZOk6WhPj99z9UsojGI3d3xI8fUn7IlD6WaOPs7IJ79+6iadPGcHN1UwHn6dNnwIiRo8KUQ4opvH0CMXj8DQzqmRUVSyVR5Wrcn/hg0F8hxbsZV8gahs7KhRkco5g2pSM2zC+irI5uLnZYsvYR1m+LudCI6HCxU1/kXzgZ5S8fRKCfv3I1n2/dw8DlnCBnNnjd17P0BwHlzu6B34tXcEyXWonD64MMayde6TUY+eb+jXLn98Pf8yUcUqXAnfEz8WzbHnwpZiy8jSF9c2H9ghLK5c9kk+ETruHRk5CZfOqUjkgYP+S3+PK1HxZOLqysj0xWovt4yF9XdWWK+DkPGhMcp7hpSUm8fuOHlMkdlRhdvcnQOxIbcC6aF3mmhrh3sw3ujgzdf4bHpj24NXrmF923HDlyomOnzhg3bqyyKr709ESZMmXRqFHIU7Dev3uvEi80i4spfUxpw5jzP/oHlyV7/OiRivu6eOGCugf91ruPWt+8RUt1He7Qvh1SpEipxBgn/P36Gc+KNUaLFj/D46kH2rRupcQEr9fde/Q0mMg+ffoE7g/do9QnsjY8njmzZ6lrJpP8uIT0bYkiRYuq/48cOYzVq1bpzu/EiRMQzyGeKixeuXIVfC6+hkLZsR2boKg+nC8acBZCAcKbN8Ubs2YpZjQLHi1+NHEbcwlSsNBNyxkO2+u7hWkZ5JjHjh1TljZavvhjoyk1dKJBROMQike6K7VZEU238+bNU89JNFVIsp4jEzs01zr3j8dNAcc6RXQRsKSPfqIIrVoUrExg4Q+PJubQ4xhrQ1jf8OLFi2q/WQST+0vLoX72E4NVGddAwUh36eLFi9X50W/D/WOwKy8AFO9MIopKosat23cN4hppyUuVKnUYt7W7+0P1OaVMmcpgfUR9LNmGrucnT56qmbkxsR6a27dvqc+eArNtP8uIMp7WNCkdVaKDZjnUfy9rxvh44uGrrI0aCZxskdjNHo+f+oRJdAkNrZDpUjuqjGYmwESFAYdMq8MW7rbTp1XJJR+YZRzqspIgZ1b4v3kH38ch55HlcRwzpIWP+xOVrRweDimTwza+E7wfuJtV5mZosemwFIwxpfv30RNv+PkbHmuqFPGUJfDhI2+DzzZtKie8fusX4VNXOC5F59NnPkZrY0ZG/x3tEdOw0HmC7GGt6n7PX8L7QXANy5gk+7Udkbbh9fvpkydKzOnHJmvX5SePHyvho3+diKiPKW14nb5z+1PNUj0c4sXThb9o8DrF61WiRM5h4u9NhUaBt2/eIGWqVGEm7QwnouEk9HYj6hNZGxpX7objWUqRMqXumvrmzWs88whbvSJxkiQ661iWzFHzykSHDSeNJ9dElbrFTH/CjbXxWQRiTKEvEENXERc+H/oC0VqxlECMzZgrEGM7lhSIsZXPIRC/NKYIRCF2IwLx60AioE2EwZ768Xz6MDuI9QQFQRAEQfjyBFmoDuK3zFctEGn+ZuKF9gSSmITp5eFVuw9dpFkQBEEQhC+HxCB+4wKR8SP68XwxCesCRveJKoIgCIIgfD6+3uC52EOMP4tZEARBEARB+Lr4qi2IgiAIgiAIoRELovmIQBQEQRAEwaoI/AqepBLbERezIAiCIAiCYIBYEAVBEARBsCrExWw+IhAFQRAEQbAqRCCaj7iYBUEQBEEQBAPEgigIgiAIglUhhbLNRwSiIAiCIAhWRZBkMZuNuJgFQRAEQRAEA8SCKAiCIAiCVSFJKuYjAlEQBEEQBKtCYhDNRwSiIAiCIAhWhVgQzUdiEAVBEARBEAQDxIIoCIIgCIJVIRZE8xGBKAiCIAiCVSExiOYjLmZBEARBEATBALEgCoIgCIJgVYiL2XxEIApmEweBVn8W541NAetnM6yZRfgW2AFr53qO6rB2cl7b9qV34asn0PpvSzGOuJgFQRAEQRAEA8SCKAiCIAiCVSEuZvMRgSgIgiAIglUhAtF8xMUsCIIgCIIgGCAWREEQBEEQrAqpg2g+IhAFQRAEQbAqgizmY7bBt4oIREEQBEEQrAqJQTQfiUEUBEEQBEEQDBALoiAIgiAIVoUUyjYfEYiCIAiCIFgV4mI2H3ExC4IgCIIgfAHu3buHli1bokCBAqhatSr+97//Rdj+2LFjyJEjR5jl7t27Zo1rDLEgCoIgCIJgVXwNZW5ev36NMmXKoHjx4pg+fTpOnDiBevXqYcOGDahZs6bRPh8+fMD169dx9epVg/WpU6c2a1xjiEAUBEEQBMGq+BpczHPmzIGXlxdWrlwJe3t7lC5dGufPn8eQIUMiFXK0GsbEuPqIi1kQBEEQBOEzs3//flSsWFGJOI0ff/wR//33H968eRNh37Jly6JYsWJo3bo1rl27ZrFx9RGBKAiCIAiCVREUGGSRxdfXF2/fvjVYuM4SPHz4EClTpjRYlyJFCvX30aNHRvvY2NgoUTh06FCMHz8eAQEByJ8/P86dO2fWuMYQF7MgCIIgCFaFpWIQR48ercSYPoMHD1bu2tAcOHAAHTt2jHC83377De3atVP/U9zpW/mIg4OD+vvx40ej/StUqIDvvvtO97pcuXK4c+eO2qdNmzZFe1xjiEAUBEEQBEEwQv/+/dGrVy+jYis0RYsWxcaNGyM8j8mSJdP9nzhxYrx48cLgfe013zOGra2tUXfzmjVrzBrXGCIQBUEQBEGwKiyVpOLg4BCuIAxN/PjxI0weMSYot27darDu6NGjyj2sn5UcGQ8ePICzs7PFx5UYREEQBEEQrIrAwCCLLDHJL7/8otzD8+bNU69v3bqFuXPnokOHDro2hw8fNqhzOGHCBFy+fFn3/rp167B69Wr8/PPPURrXFMSCKOhgbIKdnZ3KgGKcg6VhXMSixYuxd+9elYLPL32Xzp2RNm1as/p4enpix44d2L5jh/p/yZIlSJokie79oKAgHD12TBUK5Q/F0dERhQsXRquff4arq2u0jmXr1i1Yt24tXnp6Im3adPilbVsUKFDQ7D6WaOPl9QELFyxQta/evXuLJEmSoEqVqmjYqLEKcNba7Nu7F9u2bcX9+/cxePBQFCteXDeGt7c35s6ZjSNHDqvvRcFChdCpU5cI3ROm9LFUG3Lq1EmsXLEC9+7dRfLkydG8eUuULlMmzH5xjL59euP69WsGx8lCsrNmTlc1xTjzr1q1Gpq3aIk4ccKfN5vSxxJt+J09d+4stm3dihMnjqNEiZL4Y8BAg30xpU1oXr16hZkzp+O/M2cQx9YWJUuURIeOHeHkFN+sPqa0uX37NrZt24ID+/er3928+QvDbIulOJYvW4I7d+4ijm0cZMmSBa1atUa2bNnxOXFMnxrp2jZC2tYN4JA8CXa4FETABy/ENlavWYMtW7aouneZMmVCxw4dIrVgRdaHSRi7du/Gtm3b8OTJEyVIcoYa093dHf+sWoWzZ8/Cx8cHGTNmRLOmTVGoUKEYO1ZrLHOTJ08eLF26FF27dsUff/yhPpMWLVpgwIABujbv379X1wktMaZIkSJo1aqVEoy8BvCaMW7cOHTr1i1K45qCWBCFzwaFG4XeoEGDlIBxdXHBHwMGqAuMOX2WLF2KgMBAtG7VyugYzNrat28ffvrpJzXemNGjlSgKHXhsKv8ePIg5s2fjlzZtsXTZCpQsWRJDhwyGu/tDs/pYqs2smTOVcBg2bDhWrV6LDh064Z9/VmL7tm26Nls2b1Zuic5dQi4q+kyZMgmXL1/CuPF/Y9bsufjw/gOGDh2MwAgecGpKH0u1OXz4EEYMH4bvKlfG8hX/YOjQ4Th69IjR/VqyeBGcXULcL1qx2QED+iNFypRYuGgJfu8/QInlf1auCPf4TOljqTaPHz3CmjWrUa58eSWQjWFKG314MxkxYhhev3qNGTNmYcKEibh58wYm/P23WX1MHXfG9KnInCkLvv/hR6Pb4uRu6JA/kS17DixeshRz585DksRJMGhgxNeImCDHyN8Q8MEbN4ZNRWxl8+bN+Oeff9Cje3csW7oUOXPmxICBA/Hy5Uuz+qxdtw6vXr6MMNliw8aNKF6sGGZMn45FCxciX758GDxkiMqeFaJG06ZN8fTpUzWhZ5zg/PnzETduXIP4QhbFppjXklJOnTqFmzdvqvXPnz9Hz549dZN/U8eNcYFIZcud4gyjRIkSSJAgAbJmzarLpNFPr2Y7Kt306dOrgE9aCfQZPny4eo+zaY7FG7op7/GL3aZNG2UlSZQokbJ8nTlzJsw+Tp48WVUWZxvOSmmS1efSpUuqmCStS5xNzZo1Sx0PZ1pR3dakSZOUhYpxC/w/Mjhu27ZtlRWEcQRNmjSBh4eH7n3OFGrVqqX2h9tt2LAhHj9+bGCN4HZpHdMnXrx4uv035TzwuAjrJ7FtVGIpIsPPzw+bt2xB40aN1GyUFoQuXbqomc3BgwfN6vNrz55o2aIFkiZNanScNGnSYNDAgciXNy/iOzmp17QeXr9xw+A8mwqtdzxHtFYlTJgQzX5qjiRJkqqLrzl9LNWG34PiJUoiQ8aMKpOtSNGiyJQpM27duqlr06hxE3Tu0hUZMmQIs6+84Bz691+0bvML0qVLr85rp86dcfvWLVy4cN7o8ZnSx1JtaFWeM3sWfvyxhlr4m02WPDn69O0XZr/OnDmNY8eOKpGsz759e+Ht5aUsky4uLmrGXaduPWzatFGNbwxT+liqTeo0aTBq1BiULVsOcW2NX9RNaaPPlSuXce3qVXU+eb7SpEmLVq3bqPPz5MnjaPcxddy/J0zCDz/+CMd48Yxu6+HDB8pKUqdOHXWtd3Z2QY2aNdW1y8PjKT4nZ5v3wq0xs+D79DliKxRyP/7wg7rX8Hre9pdflADYvn27WX3atG6tsmzTRBCr1q1rVyVceL/idYj3LH53b4a6B31pC6Ills8BPXe0wvIzCS+uMXRWMu+H+gkvUR33s1kQR40ahZEjRypLTadOndCgQQPcuHFD9z5VLGeZ/v7+6kZGF6Z+ivj69euVkFq7dq2aRfLRMDSPRvYev5DVqlVTLiSaurl9CqnKlSurbepDYfT3338rkzlFBp9RqIks7hf7pUuXTokt7uOCBQvUTF8jKtvidqZMmaL6U9lHBMfjGJwJ7NmzR1l1+EgcLSOJ261Ro4YSbDQzX7x4Uc0G2IbnNKpEdB60LCd+Phw7dPFNc6DI5aSAM00NCt7MmTPjSqhHBpnTx1S0YqFOTk5R6kfReufObeTV2yfCOlRXr16Jdh9LtSHlK1TAiePHcf/+PdWHIon/U0iYAm/2/Pz1zzvFmourK65euRLtPpZqw98BrwXlK1SM8Djo9pw8aSJ+691HiUh9OFb27DkMgs/5zFKKkYcPHhgdz5Q+lmoTE1y9clX9fjJmzKS33eDQhPA+V1P6RGdcY/CcJE+RQoWC8NpJVyfd5zQ6UHQKIfD7/+zZM4NrAbNb8+bNG+61MTp9TIGf1bq1a5VQzB/q2vQlCQwKssjyLWMRgUiLIOvycDbB//n8Pwqk0PDLyAs//eD6KdkUZbQQMvOGVi/ObhYuXBjpe4yRoLWEYo4xafyCcvtUzAzc1IeClPtFNU7RxlkTzbSEFySKPFoNacXjBWnatGkG/aOyLfr8aY00xZxLMUrr5apVq9QPleewUaNGKnZAe5+xOzQPM/uI54L7cPLkyXAtbxER0XmISXizJvqZVoQWFO09S/Qx9YJGtzQ/I36OUYHWS7o6nRMZ7pOziwteh7NPpvSxVBvSqFFj5M6TG507dUTdOrWUC/rnVq1NckOSV69eIm5cuzBxaS7OzhF8VpH3sVSbp0+fqL+0TrX9pTXq1a2Nbl274Mjhw7r2FJl/jx+H6tW/R44cOY3s76sw3ytarLT3jB9j5H0s1SYm4LkNvV1aJRwdnSL8XCPrE51xjUER/+egwSqMolHD+mjapJGapP7xx0CjpT2+ZWLD9ZRZsd//8AMaNGyoXM5/9O8fpRIqwjciEBk0qQ/F3BW9mSPFD9vQzElLGF2k+rEKtIbRokXhwqKU+oIlovdOnz6tbpoUjryAcKEbmxY+ZvDoQ9GnwX3Q/1HwIkR3q/4Ph8G2+kHlUdlW7ty5TT537E+RSbenMXgeGXuguX8J21PI6p9jU4noPJiCpavK80Zu8xn6aNCiNnzECCWMGYdjMdRM08byfaLRZtq0KcpyM2XqNKxbvxF//jkEixYuwJ49u03etVDhLLrNhI5ziWofS7TRLOfr163Fn4OHYtnylahSpQrGjBmFSxcvBr+3fh28fbzRuEnTSI7UYCufdgAW7mOpNjGFOrkx0Cdq4zK8oH//fihbriz+WbVGxZbStfb7731VUpVg4rUxip9ldPqQUqVKqbhmPu+Xz/dlDOKt27djzccUFGiZ5VvGIgIxoi8XhRXdmLRW0RpIlymtYvrVvBkHxYBLWuSYPMBnBlatWlW5fiN6j9YUCh6OxXG5cJ2yHoQKkI5oH035gURlW6FjBSIism2H50Y2ZZ+N9Y3OhUAfinQKaf2FltfI0LKFQz8Hkq/pPrRUn8jE4bDhw5WrZfSoUZFaD3fv3q1myFx+/KE6jh09qgQ1JwZv3hrbp2ArUGhM6WOpNgzm37ljh8pYzpw5i5rQMAaxYsVK2GSkgOu4v8aqv0wC4TESV1c39fsKfVPmdrkPxjClj6XauLm5qb8UfwwLYZhArdp11P/MfCZXLgfHxdWq+aM6rmZNG+uO8/d+fdR3y9j3Su2DS/jfx8j6WKqNuXTt2lkdN5fWrVp+2q5bmO3yN8EwDiZ/GcOUPtEZ1xj79+9T16w2bdqq3yY/746dOivhSGuVYOK1MdzPMup9TIHW/RbNm6tcA14zYwv8Llli+ZaxiEAM7aLka2ZHkePHjyth1bx5c3Vh5w2O7tHQMIalcePGSmzQqsYvGrNvInqPVj66XykgzYH7Svex/g+H29HPmrTUtkLDcWmBDO/5iLRG8n39qugU2owl0c6x9sPXtwRSTPMiHRX42XCJKFOVVeV5nvSXyB4tpFk96UJiGQsNxlzxnOb6dByW6BMefv7+KtmJSSljx4zRiYyIoFWKM2QuW7ftQMlSpZT4Z8LHBb19ItzHnDlzGR3HlD6WasPPj5OA0BMBrRxCaLTEDpZ/4TGSHDlzqv7nz1/QtWNsLN3YOXMZP0ZT+liqDUue0A1t1Mr46RgH/TlYHY+2rFi5SnecY8aOU2Ox7I2+9ZvnkdeaNOGUXTKlj6XamMu0aTN0x85M6eDt5lS/H32Px/nzwc9vDe9zNaVPdMY1hm0E5YUiKj30LUJXLhMULlwI+Z3QaMFwpfCujdHpExVoPPkiBnAhxrDIr45ZzMwspljg/xSFWk0euggoqpiAwdgvJp1MnDjRoP9ff/2lYhYpajjr3Llzp3IB0iIQ0Xs0axcsWFCJR4pSjs8kDm47KvF5TDahy7Zz585KeFGA6NcUIpbaVmg4bq5cuVRKOn+oPIfMLNZiIJmgQhczs8qYTMKbJTOe6cYvX768akMrHuMXacmkG5wCMqoFMbWLMOMrjx07Fu7zGhlYz1AB/cWUKvMUNzwW1uCiS5/7yYQjzly14yB9+/XDsGHDotQnMmiRGjliBDyePTNZHEZEvfr1cfDgARXz9u7dO6xYvgzPnz9Tn6UGs2w1y42pfSzRRmUtFymK1av+we3bt5QIOX3qlLLOlCpd2qTjY+YwE1oWLVqgvm+04MyYMQ2Zs2RBvnz5de14fDxOU/tYqg3jZ+lJ+GflStWG14X/bdqosmBZAsIUKlX6Do5OTqp235s3r1VZnQ3r16F27Tq62GFeB2iB07KnTeljqTYxQa5cuZUAnzVzBp55eKjSSAsXzEfJkqWQMmUqXTse88aNG0zuY+q4kVG0WDFlAWdffrf5e+eYnCTmz18gBs7I1039+vVVbPyZ//5ToT5z581T17rq1YM9AeSvcePQXS+UxpQ+kcF+Y8eOVUYVfl7PX7xQ4zCOv2KlSogt0M5hieVbxiJXI1qVuLC6N83MFDhaiRRm6DIphRZE/uCZccmHVevXoGPpGGZB8+JOSxktYxSSFIERvUcoPJkUUrt2bWVBy549O9q3b68SEEyFqeBMVGE/jstEEP6o/vvvP5344YXbEtsKDcelRbRPnz5K9NB6xx+rVh6H7/OROdyfbNmyKRFHFzvr+elbiZYtW6b2hYksdMtzPD44PKpQ4LMvk1mYLWzJTOafW7ZEUGAghg4bpopes3QNhRvdoOb0Wbx4sSraqsGQBtKnd29UqlRJxcWc/GTl/ql5c4PxKRj1M2ZNoXz5CuoGNn/+XHh6vkTadGkxeMjQCDMtTeljqTa9fuutav8NHzZUTThYBodlbRo0aGhQR3D0qJG613S9krr16qNt23bo3qOnEn+9e/fCR39/FCpUGH369IvQkmNKH0u1afNLW3WMjFnz8fZWv1taDY0lpBiDInPkiFGYOXMGfm7ZAvETJFD1+Zo0bWZWH0u1IfXr1TGo/0fhxgnZyn9WR6mNBq8XAwf+iZkzpqNTpw4qjrpEyZLo2NGwBFBU+5g6bp/evQziprWQBtY85HeU2er8LnPS06b1z+rz5sRg2PCRnz35IffkQcjQOeRaUf31WfX3VO0OeLYt6tfVmKBWzZrqu0+DC3/nmTNlwogRIyI8V6b04f1m2vTputfac4hZZ5YJlPx+scTNzFmzlNWYAj5rliwY99dfyJ4tG2IL37p72BLYBJlxFulWYKwILU6sT2hNUOyyPhmtn0xgEcLnTiwKTI4pAqWmvCDECq7nMN3a9bWS81pIUXtrJFPmzDG+jT8XRy3EKjyG/Wx6ToG1IY/a+8T48eNVLTIKXbqzNcugiENBEARB+LqI4ccofxNI5K9eHCLdqyw3w4LYdOduNJL1GR3hqSUNhF5ooRQEQRAEwbIEBQZZZPmWMcvFLAhEXMyCIHwuxMX89fM5XMx/zI9+fV59Rv0SeRKmtSIWREEQBEEQBMEAiUEUBEEQBMGqCPzG3cOWQASiIAiCIAhWhUTPmY+4mAVBEARBEAQDxIIoCIIgCIJVEfSNPwXFEohAFARBEATBqgiUAi1mIy5mQRAEQRAEwQCxIAqCIAiCYFVIkor5iEAUBEEQBMGqkDI35iMuZkEQBEEQBMEAsSAKgiAIgmBVSI6K+YhAFARBEATBqgiSJ6mYjQhEQRAEQRCsCilzYz4SgygIgiAIgiAYIBZEQRAEQRCsCnExm48IREEQBEEQrAoRiOYjLmZBEARBEATBALEgCoIgCIJgVUgSs/mIQBQEQRAEwaoQF7P5iEAUzCbgG/gaxbEJgLUTFGTzpXdBECIl57VtVn+Wrub4AdZMJv/rX3oXBBOw/ju7IAiCIAjfFEHyKBWzEYEoCIIgCIJVEShBiGYjWcyCIAiCIAiCAWJBFARBEATBqviaXMw+Pj64d+8ekiRJopaIePDgAV6+fBlmva2tLfLmzav+f//+PW7duhWmTc6cOeHg4GDyfolAFARBEATBqvhaspgXL16M7t27I1GiRHj27BkaNmyIBQsWwN7e3mj7efPm4X//+5/Bups3byJFihS4ffu2en38+HFUqVIF+fPnN2i3ceNGZMiQweR9E4EoCIIgCIJV8TUIxPPnz6NNmzZKJDZv3hz3799HiRIlMHz4cLUYY9iwYWrRoLUwZcqUaNmyZZi2586dM2v/JAZREARBEAThMzN//nxky5ZNiUOSPn16tG3bFnPnzjV5jFWrVsHLy0sJzdA8evQId+7cQUBA9Mq0iUAUBEEQBMGqCAwKssgSk5w5cwbFihUzWFeyZEl4eHgocWeqyKxevTrSpk0b5r2CBQuiTJkycHZ2xqhRo6K8f+JiFgRBEATBqrCUi9nX11ct+jDRw1iyx9u3b5XFLiJSpUqFZMmSqf89PT1RvHhxg/cTJ06s/r548QKpU6eOcKwrV67g2LFj2LBhg8F6Jrrs3LkTVatWVa+3bt2KunXrqvXt27eHqYhAFARBEARBMMLo0aMxdOhQg3WDBw/GkCFDjMYUduvWLcLz2KNHD7Ru3Vr9b2dnF0Z8aq/5ninWQyan1KhRw2B9gQL/b+9MwGaq2z9+0/JW1jdFSQvJFmVXdkVCiexZQopS0iLJG1rsZMuafcm+lBRZE8ruJUuWsoWU7C9Szv/6/PzPNDNmnmeepZ75nbk/13WuZ56zzJwz58w59+9evnfBgP+rVasm9erVkwkTJqiBqCiKoihK7JJcMjcdOnSQV199NWBeOKmYMmXKJKgwhLDw4cOHA+YdOnTI/M2WLVuc2168eNEYfOQsXn11/L4+vJFff/21JATNQVQURVEUxXOdVJJj+te//mUkaPynhGgJxsVDDz0kS5YsMTqILp999pkUK1bMfA6cPn3aGJ3BnkakbghDP/PMM1e8b/C6GMtfffWV5MmTJ0H7pyFmRVEURVGUf5jnnntOPvzwQ6N92LZtW1m9erVMmTLFGIkuK1eulCpVqsj27dsDDDzCyxiYd9999xXv27JlS+OdLFeunDEOqYrGyMQYTQjqQVQURVEUxXNFKskx/Z3gJVyxYoXRMXzrrbdMwQkFJVQl+6+D4PV1113nm3fy5ElT6YzAdiiGDh0qGTNmlD59+pgcyixZspiCFiqkE0Iqx6Z+NEpUsmvPPvE6qVMlTkfKJhwnVUrvgqLES2q55PlvaXuequJlql38/m//jKfePJgs7/Nxj7hzAb2MhpgVRVEURfEUziXvDyT+bjTErCiKoiiKogSgHkRFURRFUTwFFchK0ogpDyIClbVq1Yp4fcrHn376aUkpzp49a/SNqGzyIr///rvs3fujHP/tt2Tdhr6Te/bslqM//xx2HWQF9u/fJ2dOn5akQsLw3r175dy5c8m6TSTrkKjMdCmOcAoyCTt37jTXU1Kh5+feH380HQOSc5tI1jlx4oRpP/XHH3/E8T5nzXnl/RICmmL79u01shHJuU0k63Du6L4QrIcGbLdr586QU2LTx9nu4MEDcvjwoWTdJpJ1+P3u378/3mvxwIED5tpPKvzOeZ8TJ08m6zZcg7t375Zf4jiv/G73RXCsKU26Arnlhpx3itfgekyOKZaJKQ8ihkNCmlYndP3khouTz/fiRbp0ySIZMvhDSZcunRw//psULVZcXm/3Zpz6UvFtg1HwyZxZsvDLBXLy5AkpV66CtGkbKHD666+/yJhRI2Xd+rWS6cZM8vPPRyR/gfvktdfb+3SnIoVzM2jgQCMdQAuj48ePy9NNm0qNGjWStE0k68z99FOZMWOGpEqVyqzP3xdat5YHHnjAt87+fftkxsyZsm7tWmNcde7S5Yq2TglhxvTpMmnSBLnxxkzGcKGN0/MvtJbUqVMnaZv41vn2229k4sQJZlBwww03GMO5YcPG8oTf94FROGrkSNm583vJkCGjOa+lSpWWNi+3lWuvvTbO41q1aqUMHNDfVAlioN6bP7906PCW3HBDmiRtE986aJV98skcWTB/vpw4cVyKFCkqb3X8T8DnfPPNKlm0cGHAvMNHDnNzkMlTpslVV10lCYGBU7euXeX8+XNy8Y8/5OabbpKO/+lk2n8lZZv41mGwNm78WFn97WrJdFMm+eXoUSlcuIi83PYV83t2WbBgvsyb95kc+uknU4U5ctQYSSyfzZsnI0eOlH//+9/muipbpoy88sorcYoKx7cNg62Zs2bJ4sWLzXX4+OOPy7MtWgS8x6HDh40EyZYtW8z7HDlyRIoVLWrElrl+o4FU11wjd7VuJHc8W0+uy5pZflu5QdY+FngcihJTHsSBAwfKrFmzUno3Yh4e5v0+6CMtW70go8dOkFFjxsvuXTtl/LgxSdoG4w9jqWfvvnJPrtwh34cHT9ny5WXylBkyZNhHMnrsRDly+LCMGf1Rgs8L19K3334rw4YPl9FjxsibHTrIRyNGmAdDUraJZJ2jR49K3759Zey4cTJh4kSpWrWq9Oje3XgTXXbt3i358+eXDwcPlqSyYcN6GTdujHTq3EVGjR4jgz78UJYtWyqfzf00SdtEsg6G7muvvS6TPp4iH40cLa++9rqMHDlC1q9f51vnwP4DUqdOXWM0DRs+QoYOu7x85ozpcR4X32PvXj2lfoOnZOy4CTJu/ET5+cgRGTF8eJK2iWQdDAy8S+937SYFCxYK+VmPP15dBgwc5Js+6Ndf/nXttVK+fIUEG4d4M7u+/565JiZOmiyTJ0+Vm266WXr06JakbSJZZ/+BA1K0aDGZOm26DB/+kYweM04OHjwoQ4cEXpt7du+Wl156WWrWfFKSwo4dO2TIkCHyRrt2Mmb0aPP7Wb9hg0ydNi1J22D8MSAd0L+/0ZkLBcdVrWpVmTpligwfNkxGjxol3+/cKZMmTZJo4ZoMaY1huK5GK/n508XiRWyQuYl2PGMgEg6uXbu2vPDCC3LnnXfKjTfeKM8//3yAojgjQQQp/eFHW6hQIUmbNq0ULlxYPv/887CfsWbNGqMn1LNnT9m3b58ZVe7ZsydgHd7HfQ83RIxhWrp0aTMizps3b4AIZiSsW7dOypcvb44JocyZM2cGLGdfnnzySfP+rNOgQQMzavVfHum+ItqJlylNmjRmv/8OFi1cIDdnziwVK11uJI7nqErVx2TRoi/Demwj2eaOO+6URo2flptvvtwIPRT33V9QSpR40OedypAhg9yTK5d5oCcUPD8PPfywz0vC98Z70SQ9KdtEss4zLVrITTff7PsfbxrhO0LJLg8//LDxxsXnQYuE+fO/kPz5C0ihQoV933XZcuWMxycp20SyTt169SV79hy+/x944EHzO/zuu+9880qVLi35CxTw/U9/UlpVxXdelyxZLNddd70xxABv1hM1aspXXy0L6G6Q0G0iWSdz5szSrFlzo4EWKevWrZXffvtNKj9aRRLKxo0bzACiYcNGxuOMgVm/QQNjlO3evSvR20SyTtGiRaVChYd8Ri33Ku5p330XOJh6ofWLcs8990hS4beSI0cOKVmypO96qFSpUty/zQi2yZ0rlzSoX9/cZ8NRvFgx8yzhu4BMmTJJjuzZ5edE3GP+Ln7/9bhsf6OnnN2V9DB+tKIGYtLxjIGIkYDhRINrDLn58+fLF198IR07dgxYx98AIZTw7LPPmubaGFFjxowJO8pbtGiRuVm899570r59+7DhX3JT3Hwwdx2aer/99tvGQKNJd82aNWXXrtA35FDgKeratav8+OOPxhBu1KiRL1+J96dRN8YBjcLXrl1rejliMLokZF8R1ezVq5ccO3Ys3qbjiYXcnVz3BHr4cufOI2fPnDGhweTaJi5++GGPbN+2VT79ZLasX7dWnqwVOHCIj/+dPWu+51y5cgXMz5M7t3kwJnabxLyvOZ7/N/6zZA5vHCeFPbv3GCM1+Psnn+zixd8TvU1i3pf0Aa5PjER/uJbJzdu69Tv5+ONJJl/xsf830MIf127JmfPuAG9c7ty5ze/pwP79id4mMe8bCV8uWGAMqFDdE+KDfcIwy+z3veXKldsYMuGurUi2Scz7AnmXmTMHnsPkYveePSF/Q7/88ovx3CbXNnHBPR5xYlJBdnz/vdSMI/VEUaIRzxiIbjNqjCkeHMWLF5du3brJ4MGDw3oC3nnnHZMX0rx5czPKQ608lIE4ffp0Y3BhUNIaJ6G89tprUrlyZfMZb7zxhumzmBDvHAZmqVKljLfr9ddfNw8dDEHAG4nXCOMWzykPjtGjR5sQJb0XEwoGNe15yJtyR8DJzenTpyRd+r/yjiB9hgyXl4UpUkjMNnExauQIGTSwv4wZPVIefLCU3HvvvQna/vSZM5f3wS9/yvyfPn3YQotItknM+5J7iecXT3iu3KFD60mF7z99uvRX7BNG2ZkzZxO9TULfl4HMwAED5N833ijlypUPWEao88MPB5m8v2lTp0jlyo/KXXfdFe9xpbvi8y9fV6dOn0r0Nol53/ggF3Xt2jWJ8h5e3qfTV+wT9xKiBafCFGtFsk1i3nflyhUmR7NuvXryd8BvJdRvyN3f5NomLgYPGSL9BwyQ8RMmSIXy5a8YCCl/L5ecS8kyxTKeKlLBre+fgExoDuMQzxuhXX8IPZErgjEUFxhZ9EacO3eu6YeYGDAI/WG/Nm/eHPH2/qNajDZG6zwsYOvWrSYscrNfuBEjkfAVo9f4ji8Y8ojigpB9cCPw3y9ckGtDFJfgkXQ9W5A+Q3rJkuUWufqqq83DPPg94KowCeSJ2SYuunbraf7yPXbp1FH69O4p/3m7S8Tbu56h4H268PvvYZPgI9kmoe/LuXj33XfN6/ZvvilJhfcj58/lxkyZzMDmKvP9B3r0fr9w+f/wxxv/Ngl9X4qUtm3bKt179JLrr78+YJnJDRs4yLymkrbDm+3N9/bccy3DHu9Voa6r3y/4rrnEbpOY942PxYsWmQhJsGEcLmf3wvnLn8c2d2XPLlddfdUV+wTMuzpMPmMk2yT0fYl09O3TW5o83VSKF0980VRccO1c+f1fvq7C5W4mZpu46N+vn/lLpXOHDh3MQA6HhPLPEOv5g8mBpzyIwR4v9/+4JEDi85IRzsELMWHChIgqmkNVHIfar7j2KZJ99P+cUMuZF1/1c6jl8eWrEYLGk+k/DRs2JOS63FwHfzjANy1edLkakxzB344dC1j32G+X/w+XP5iYbSKBKsNHq1SVdWvXJOicsB0PlGNBcjvso7+xntBtEvK+fL94wU8cPy7de/RIcBV2KH795RfjhXOnNWsuSyzdnPlmk/sW/P1jpJHLGopItknI+w4dOkRWrPhaunbrEa9n8NZbs5pCjjWrv41zvcufH3RdHbu8P+S8JnabxLxvfCxcuEDKli0XUSUsBRLuOaR62HzuzZlNtbT/fQzPGIOCsMcawTYJed8tWzbLu+90lrp165np74LfSvBviLQEco8Z8CTXNhHty003mVzg1WvWJPo9FCUl8JSBuGnTpoCbFGFYvAp42ILBw0YRAI2y44J1li1bZvIaGzdu7Ht/HuSAfIi/dpc74gwuMvGH/Qr2aCYW3odcHv+HLPlbJI27n5GQfY0PRsLk4/hPrVq9EHJdHvL9Bw72TU81bGzm31+woElO99eqW7N6tdyVPYcxOIFl5JO53spItomEUMfMd0U4LC65lmDwKBQoUEDW+t30yencsGGDFCxY0DeP/CU82JFuE+n7GuOwSxc59uuvxjjEq5wc3JYtW0DVbJUql3vCFry/oKkK9v998f3fd//9vgEKITrOmbtOJNtEsg4wCPlq2VJ5v2v3kL/nkOf16M8BEiqhKHh/IZMr5v/bwKjk/uAWj3ANclzutRfJNpGskxDIqyTiEWl4ud0b7X3nsHPnd8w8UmiIqHznVw3PAIBrjkIhF47V3e9Iton0fSkq6tK5k9SuXcdUd/+d8FvZuHFjgEcQAy1fvny+QfCZM2dMeo67TiTbJOUeky7MQEr5e9AilaTjKQMRwwgDhgcVIVzy6cgZDA5FubAuOYtTp041Vbzff/+9KQIJhmrIpUuXGsHqJk2amIcZRgl5a/369TM3Gm7erVq1Cvk5H3zwgSxfvtx8BrmH33zzTbIVgKDDhTeF4yRsTmJ+ixYtpEiRIqZKEBKyr/GBwY2nyn8KFV6Oi4crPiKZMt0k3d5/xxgH06ZOliWLF0rjJn+JklP9+ErbF+Wngwcj3gZcEeFz5/4np06dNK8R1nYZNuRDGTtmlPEY/nfTRpk0cbzMmT1T6tZL+AOrUePGsn79epP/iQHXo0cPY9Q8Xv2vwohZM2caQy4h28S3Dp5fiqUwQKhmxgjlQcfkb5C4AtkMIODwoUPmf7yECaVGzZrGCOjTp5ds3LBBPhoxXLZv3yYNGjT0rbN69bfStm0bn7B3JNtEss7o0aPk83nzTMXvpT//9J1j/+Kkbl3flylTJpuKWt5n+LChsuLrr6VOPF6q0mXKyJ133iVd33/X6EXOmTNb5s791FxXroHK74XjcityI9kmknVc/UCO5ezZM+a3ucvvfPlDVTe/c1QMEgv7U75CBRkwoJ+sXLFCli5dIiM/GmEq4P0HGBwrUkORbhPJOuhTdun8thQpWtTI3fgLfvuD0Dbz8ORhqLnrxCWOHgpkZjDqunXvbqRqxo8fb+7fDPJdeEa83Latb3AdyTbc+93fGgMHvPe85tnjwjMF6Sl+wxs3bTJSVRRN1vub8i0TS7r7ckuGIvnlmn9nkGvSpzGv0xfKJ15BhbKTTirHIyrMTZs2NT90QgFIt/BwRNJm6NChvpDMiy++aG72c+bM8W03fPhw84PGy0OVIQ9eqozd9+TGNHHiRJ/XDaPrwQcflHHjxpkbDFXQjIwpEGnXrp35DKrWqCzmho8HA1kcjFDyBZFOoEq4bt268R6Tuz0Gpb8AMgbr+++/b/YPqI7mczFi8YIRzqBogfVcGBlHsq/BnxUJu/b8lbMWKeT+TZ3ysfywZ7ekS59eqlZ9TIoU/StXE43DDwcNkHZvdDBerUi24eb92ittrvgs5GDc/EIeOvPnfy4bN6yX8+fOSRYjZfGo5IunSCV1qtDpBZzTT+bMMQ80dNHq169vzrG/puGWzZuNSHWk28S3DsdA4VMoKKZyBwZ41ClYCuahhx4KKebtOHGnW1BdTfEHvyEEvGvUfDLAYMHzN2nSROnR86/8wPi2iWQdjD9/fUf/XN6nGjYyr/HuoZ245bstcunPS6ZgrWrVaib3Lj64V0ydMlm+/36HpEmTVh6pXFlKlizlW84gpVevnvLiS218EizxbRPpOm+0e/2KnN40adNIt249fP9zD+r4Vgep9MgjUrFiJUkKXDuzZ8+SDevXSeqrrpIHHywp1ao9FuA9f7nNS1LTXEcVIt4mvnUWLVpoxN1D4eaNmtf9+4U0kN959z2fsZlaIksFYeA0ecoU2bd3r4miVH/iCbnPTwpp85YtMvKjj0wRoCtbE982DPJxLARzxx13mAJC4HwiuE2uJd5EolBVHn00QfI92/Nc9tz/nRT7bKRce9Pl6JLLpQu/yzflGvztn13t4vd/+2c83nJ7srzP3OHJE+2zEU8ZiP7GXCjcHLNIQ4mh1nflYMIl5rOM9fESJMXocuGYgj/L/zOSQvD7hPqsv8tAtI1wBqKXiM9AVJRoIFID0Wb+CQMxJVED0Q48VcUcHwnJMQu3PsZUXEZUYqrd4iLUZyXXZwS/T2KMQ0VRFEWJNrSKOel4KgfRNshNxCgLNfkXJCiKoiiKEjmOcylZpljGMyHmhIaP/yniCtu64epQuC2rbEBDzN5AQ8yKDWiI2X7+iRBztRZ/teJMCvNGxq0N7GU8E1OMNsMwkrBtfOFqRVEURVESjoaYk45aJ4qiKIqieAo1EJNOdLrdFEVRFEVRlBRDPYiKoiiKoniKSzFeYJIcqIGoKIqiKIqn0BBz0tEQs6IoiqIoihKAehAVRVEURfEUzv9L3ymJRw1ERVEURVE8hYaYk44aiIqiKIqieIpY74KSHGgOoqIoiqIoihKAehAVRVEURfEUly55ootwiqIGoqIoiqIonkKLVJKOhpgVRVEURVGUANSDqCiKoiiKp9Aq5qSjBqKiKIqiKJ5Cq5iTjoaYFUVRFEVRlADUg6goiqIoiqfQEHPSUQNRURRFURRPoVXMSUdDzIqiKIqiKEoAqRzHUTVJxSouXLgg3bt3lw4dOsi//vUv8RpePz7QY7QfPYfeIBbOo5I41EBUrOPUqVOSIUMGOXnypKRPn168htePD/QY7UfPoTeIhfOoJA4NMSuKoiiKoigBqIGoKIqiKIqiBKAGoqIoiqIoihKAGoiKdZBI3blzZ88mVHv9+ECP0X70HHqDWDiPSuLQIhVFURRFURQlAPUgKoqiKIqiKAGogagoiqIoiqIEoAaiEvVMmjRJ3n///QQvUxRFSQj//e9/pX79+gleZhM//vijfPrppym9G4oFqIGoRD3Hjx+XI0eOhFz2yy+/yLFjx8QL/P777zJx4kTT0WDnzp1m3pYtW+To0aPiFQ4ePCiDBg2St956yzdv6dKlcunSJfEKq1atkm7duplzCVyfmzZtEq/wxx9/yJQpU6Rjx47GaIIdO3bIoUOHxHbOnTsne/fuDbns9OnT8tNPP4nt7Nq1S4YMGZLSu6FYwNUpvQOKEg5uxtzMmHi9bNmygOX/+9//ZMaMGVKnTh3rv8QzZ87Igw8+aB5CTA8//LDkypVL9uzZI++++65Mnz5dbGfFihVStWpVyZkzpzGAMaJg3Lhxxrho2LCh2E6XLl2kT58+ctttt0mxYsWkUaNGcv3110u9evXk66+/lsyZM4vtbdnKlStnDP2LFy9KoUKF5P777zcDuDZt2siXX34ptnYT2bBhg2zbts28Dr7XcKwTJkyQu+66S2wnX7585jjPnz8v1113XUrvjhLN0ItZUaKRQYMG0Sc87HTDDTc4FStWdI4fP+7YznvvvedUr17duXjxojmmhQsX+pZlyZLF+emnnxzbKVKkiDNixAjnzJkzTpo0aXzzV65c6RQvXtyxnb179zoZM2Z0du/e7UycONFp2LChb1m7du2cbt26OV74TVaoUME5f/68U7t2bWf69Om+Zffcc4+zbds2x0a++eabOO8111xzjVOwYEFny5Ytju1wv6xVq5ZTvnx5c/5WrVpljt+dduzYkdK7qEQJ6kFUopbWrVtLq1atZNSoUcaL2KNHj4DlV1/tnct3/fr10rRpU3NMqVKlCliWPXt22b17t2TNmlVs5c8//5StW7dKs2bNTCjdn7vvvtucX9shjFy2bFlzPKtXrw5YFmqerddp48aNjWZe8HXqnse8efOKbTzwwAPGS8jxETqfP39+wPKrrrrqiuO12ZM/c+ZM8zrYUwrVqlWTzz77LAX2TIk2vPOEVTwHN2QMppYtW4rXufbaa+XkyZNXzMeYwji86aabxGZSp07tC6VzrP5gONp+fHGdw1g4RgYA5CHafIzca0qUKCGLFi0SL1OlShWTxhIOLw28laShV4JiDXhgyMkL9kDhseDGbjOPP/649OrVy+TouZ4KcqFeeuklufHGG630yvjDMeGZwDvTs2dP33xyoTjGGjVqiO2ULl3a5Bx+/vnnAd4m8keHDx8uS5YsEdvhOm3btq3Url3bd4zkArdr18544Mi79ALbt2+X7777Ts6ePRsw/+abbzbXsc3gDU2bNm1K74ZiAWogKlEP3glGvYzsuUFfc801AcsJW9puIFKg8dVXX8ntt99uRvDff/+9SfznRv7FF194IrxF9XKlSpXkjjvuMNWihCSR3KA4p1OnTmI76dKlM4UMFKRwDvGacr3++uuv5vhKlSoltvPYY4+ZqnPSHihw+Oabb0zIGc8i0inBv00beeONN6R3797y73//W2644YaAZYULF7beQEQVYc2aNWGXZ8mSxTOGvpI0tNWeEvVgID333HOmCtQLVYRxsXbtWmMIEwLiIYynhgeVV8DL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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(correlation, annot=True, cmap='coolwarm')\n",
    "plt.title('Correlation heatmap')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "4e0c0971-7ea8-486a-a29a-b20d111f5484",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df.drop(['fare_amount'], axis=1)\n",
    "y=df['fare_amount']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "d6a233a6-97f1-4d3a-a5fc-e122f4ee11da",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "92c1cfb3-2d07-4026-a46f-50859dd87c07",
   "metadata": {},
   "outputs": [],
   "source": [
    "linear_model=LinearRegression()\n",
    "Ridge_model=Ridge(alpha=1.0)\n",
    "Lasso_model=Lasso(alpha=0.1)\n",
    "\n",
    "linear_model.fit(x_train, y_train)\n",
    "Ridge_model.fit(x_train, y_train)\n",
    "Lasso_model.fit(x_train, y_train)\n",
    "\n",
    "y_pred=linear_model.predict(x_test)\n",
    "y_pred_rigde=Ridge_model.predict(x_test)\n",
    "y_pred_lasso=Lasso_model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "c9d92c56-e9dd-4b93-b1be-78e2b5c54b6f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " linear regression - r2 score:-0.0000, rmse score:4.1448\n",
      " Ridge regression - r2 score:-0.0000, rmse score:4.1448\n",
      " Lasso regression - r2 score:-0.0001, rmse score:4.1449\n"
     ]
    }
   ],
   "source": [
    "def model_evaluation(y_true, y_pred, model_name):\n",
    "    r2= r2_score(y_true, y_pred)\n",
    "    rmse=np.sqrt(mean_squared_error(y_true, y_pred))\n",
    "    print(f\" {model_name} - r2 score:{r2:.4f}, rmse score:{rmse:.4f}\")\n",
    "\n",
    "model_evaluation(y_test, y_pred, 'linear regression')\n",
    "model_evaluation(y_test, y_pred_rigde, 'Ridge regression')\n",
    "model_evaluation(y_test, y_pred_lasso, 'Lasso regression')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98e601f5-ef6f-4981-8fb8-cf980c553d9a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8fcdc477-014c-4788-a077-26b55255d2f4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "9eaa4ecd-e491-4f77-9c1e-6ef07e53fd3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.datasets import load_digits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "460c83d5-c764-4f46-8411-159bf563595c",
   "metadata": {},
   "outputs": [],
   "source": [
    "digits=load_digits()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "aa6d53c5-07ff-41a3-82cc-7ae6cfe55fee",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=digits.data\n",
    "y=digits.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "279206e8-643a-49ce-a49a-41fa69c0fac4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1797,)"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape\n",
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "c449beba-1977-4a20-9f43-545a9c4cce81",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.20, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "1e3bb9cd-d2e7-443e-ba28-53979a2d34fc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1437"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape[0]\n",
    "y_train.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "6fd91bdd-bf71-4c16-9af8-8a243b936111",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf=SVC(kernel='rbf')\n",
    "clf.fit(x_train, y_train)\n",
    "y_pred=clf.predict(x_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "554f89d3-b4a1-4c1d-aa23-a7ca8f615f55",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9861111111111112\n"
     ]
    }
   ],
   "source": [
    "accuracy=accuracy_score(y_test, y_pred)\n",
    "print(accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "442a3f78-e3f3-4563-ac45-fc0a6e458a9b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[33  0  0  0  0  0  0  0  0  0]\n",
      " [ 0 28  0  0  0  0  0  0  0  0]\n",
      " [ 0  0 33  0  0  0  0  0  0  0]\n",
      " [ 0  0  0 34  0  0  0  0  0  0]\n",
      " [ 0  0  0  0 46  0  0  0  0  0]\n",
      " [ 0  0  0  0  0 46  1  0  0  0]\n",
      " [ 0  0  0  0  0  0 35  0  0  0]\n",
      " [ 0  0  0  0  0  0  0 33  0  1]\n",
      " [ 0  0  0  0  0  0  0  0 29  1]\n",
      " [ 0  0  0  0  0  1  0  1  0 38]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "cm=metrics.confusion_matrix(y_test, y_pred)\n",
    "print(cm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "c4d5a5e3-0652-4d15-9759-8bafb835d2c6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00        33\n",
      "           1       1.00      1.00      1.00        28\n",
      "           2       1.00      1.00      1.00        33\n",
      "           3       1.00      1.00      1.00        34\n",
      "           4       1.00      1.00      1.00        46\n",
      "           5       0.98      0.98      0.98        47\n",
      "           6       0.97      1.00      0.99        35\n",
      "           7       0.97      0.97      0.97        34\n",
      "           8       1.00      0.97      0.98        30\n",
      "           9       0.95      0.95      0.95        40\n",
      "\n",
      "    accuracy                           0.99       360\n",
      "   macro avg       0.99      0.99      0.99       360\n",
      "weighted avg       0.99      0.99      0.99       360\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "cr=metrics.classification_report(y_test, y_pred)\n",
    "print(cr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "e9499f0c-260e-43f8-96e8-850002a45103",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABKYAAAI2CAYAAABqqICfAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvctoD+AAAAAlwSFlzAAAPYQAAD2EBqD+naQAAUDNJREFUeJzt3QecXFXdP/4bCE0QQpUm2YBIh9A7JCAdJCCiIJpgAawsP3hEFEmQBwUsRFFBEQmiFGkBVHikJAhIh0T6A0gA6S2RXu//9T3Pf9bJZrbv3rO7eb9fr4XN7MycO3fu9965nznn3CFlWZYFAAAAAFRsnqobBAAAAIAgmAIAAAAgC8EUAAAAAFkIpgAAAADIQjAFAAAAQBaCKQAAAACyEEwBAAAAkIVgCgAAAIAsBFMAUKGXX365uOeee4p33nmnXy7D+++/X8yYMSP9/ZVXXileeuml9Pu7777br5azv4n19vjjj7est4Ho1VdfTcv/2muv9Wk7A+l9zfk655b1BACCKQAGjDjhf+SRR4o333xzjr89/PDDxQMPPNCpE+8XX3wx/fvf//53+nf8NHrO8Nxzz7XcpzdcdNFFxTrrrFM8+eSTvfJ8vbkMU6ZMKZZffvli6623Lj796U8Xt912W3HOOeek+z7zzDN9siztBV/9YV11xpVXXlkst9xyxZZbbtmy3hqpbW+zZs1q+PdYx7lCwHDDDTek9d3W8veWRu9re9tBznC0pn5fET//+7//Wzz77LNFWZZtPqan229X1xMADFSCKQD6vd/97nfFaqutViyzzDLFjjvuWCy99NLF+uuvX5x66qktJ2g//OEPizXWWKN48MEH23yeE088MZ3oPfTQQ+nff/3rX9O/4+c3v/lNw8fst99+LfcZLJZYYolirbXWKuaff/7Zbv/CF75Q7LDDDsUTTzyRTn632267Yskll0z3nW+++fpkWdoLvtpazv4m1ltsl/XrrZHa9nbFFVc0/PsJJ5zQpyFgf9HofW1vO+jrcLQzau/dLrvsksLHPffcM/172LBhxR577NHwPe3p9tvV9QQAA5VgCoB+7Ve/+lUxduzYdCIYvQWix1T0VNhnn32KQw89tHjhhRfS/b70pS+l///2t79t+DzvvfdeMWnSpGLttdcuNttss9n+ttRSS6W/tfbYY4+lXkTx98Fk7733TgFK9I6qef7554tHH3202HzzzecI5uK+H/rQh/rFcvY3ERA89dRTc6w3Bvb72pYIwGPZ77///tSbcvr06cVKK61U7LrrrsXXv/71Xn2dA3k9AUBXCKYA6NeiJ1P0kDrppJOKhRZaKN32gQ98oPjOd75TXHrppS29CTbaaKNi5MiRxVlnndVwmEsMt/rXv/7VEmDV23///Ys77rhjjuF6EVZFKBW9Ybpq5syZaXjh22+/3akhirUhQvfee28KiNobqtPekMbO3Kf13DURrNxyyy3p9wj66ocs1X7aWp62XmdnXlOEOk8//XT6PXq61e5fm+Ooozl2YohTtP3WW2/N8bcYrhmPjUAyRIgQy9De0Ku2tNVOrLfasLfaemuvx153dHbb6M7rjfUa20i8h43E+9Bozqn77rsvDWWrF88R962to9bLE+snHtfofW1vO+hoG6kXwXUsVwy764310xlNTU3FL37xi+LII48sfv7znxfnn39+y9/a235br/tG83t1ZT11df8AAP1KCQD92Prrr1+usMIKnbrvz3/+8zjLLC+55JI5/rbXXnuVCyywQPniiy+23HbBBRek+1966aXlhz/84fLwww9v+dv7779fjhgxomxubi7Hjh2b7tcZN954Y7nZZpuV8847b7nSSiuViyyySHr8Cy+8kP5++umnp+d69NFHWx5z3XXXlWuttVb6WXPNNcullloqPe6II45Iy1HzwAMPlFtttVW54IILlqusskq56KKLljvssEN5zz33dOk+rZfhmGOOKVddddV02/LLL9+yLPGz3HLLpdufeOKJLr3OzrymH/3oRy3Pv9pqq7Xc/6abbmpzXYULL7wwvTcLLbRQueyyy6bX+pWvfKV84403Wu7zwx/+MD32oYceKnfbbbdyxRVXTO9/PO62227r1HvZUTuN1tu2227b5vPVtrdzzz234d8PPfTQOdZ1Z7eNrr7eM844o1xiiSXSc8Vr+/znP5/qIJ5jypQp6T5PPvlk+nfUVc0jjzySbhs6dGg5a9asltuPOuqo1N7rr78+2/Lcf//95U477ZTWz/zzz9/wfW1vO+hoGwl///vfy0022aScb775yqamptTOmDFjyueff77b66er793LL7+c2t16661bbmtr+/31r39dLr744uUHP/jB9NoarfuurqfO1j4A9EeCKQD6te9+97vpRCxO2p966ql27ztz5swUIuy+++6z3f7ss8+mk9b999+/4cnm5ZdfXn77299OJ+jvvPNO+tu1116b/jZ9+vROB1O33HJLOtnde++9WwKweL6zzjqrw7CltT/96U/ptZxyyiktt2200UbpxPPf//53+ncEE9dcc0159tlnd+k+jZYhAoS4Lf5WL9pvHZZ05nV29jU1ev72lvPKK68shwwZUh5yyCHl22+/nW676qqrUsCyzz77zBFExG133nlnuu2ll14q11133fRTH+o00tl22lpvjdS2t1i2u+++e46fAw44oM110dF67Mrrje097vutb32rfO+999Jt559/frnFFlvMEY5EGPbxj3+85d+nnXZaudhii6Vgqj4Aju1u++23n2N54rG14Of6669v831tbzto72+33357CmL222+/lqAs9hMRaEdYVXt9Pd0eOgqmwoYbbpjCqdpzNXqdkydPTrd95zvfaVm2iy++ONVsR8FUR+uiM7UPAP2RoXwA9GtHH310cdhhhxVnnHFGmmtl1VVXLT71qU8VZ5555hxDqxZbbLHik5/8ZJqIOIZZ1Q/Ji+EwjYbx1YwbNy4NlYkhfyGef4MNNijWXXfdTi/rt7/97WLxxRdPk7XHxMVh6NChxec+97k55rVqJJYxhhfFkK3hw4cXG264YXHxxRe3/D2G7WyyySbFBz/4wfTvIUOGpIm2DzjggC7dp6e68jo7ek1d9b3vfa9YYYUVip/97GctE7J/7GMfK5qbm4sLL7xwjuGYMQwzJsoPscxxv3/84x9pjqDebKcrfvrTn6YJtFv/tDUpelfWY2de7/e///1UR8cff3wxzzz/91Fw3333LVZeeeU5ni8mw4951mrDB6+66qp026abbpp+rw2hu/POO9PtrcW2F8Nsw1ZbbVX0thjSG68z9g+LLrpoui2ukBjv26233tqyjD3dHjoj5mGLIa1tDY2sTXAfF3I47rjjWtb9Xnvtld6Pnqqi9gGgLwimAOjXYg6pn/zkJ2ly7r/97W8ppIqTvy9+8Yvp6lT1AVSI22sTndfEhOgf+chHim233bbNduLEcIsttkiPizla4lLtEVZ11vvvv5+WL9pYeOGFu/QaY06caCuCtTixjIndI6i4++67ixkzZrTcL078Yx6br3zlK8Vf/vKXYtasWXM8V2fu0xOdfZ2dfU1dEe9rhA1bb731HFcJjNAo3HTT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AADIQjAFAAAAQBaCKQAAAACyEEwBAAAAkIVgCgAAAIAsBFMAAAAAZCGYAgAAACALwRQAAAAAWQimAAAAAMhCMAUAAABAFnNVMLX//vsXkyZNavn3K6+8Uuyzzz7Fn//856I/Lyd9Y+rUqen9f+aZZ6zifkrNUu/6669PNfvUU09ZMf2UmqWemu3/1Cz11Gz/p2YZrDU7tMqV9tOf/rTl3/POO2+x1FJLFdtss03xiU98ohg6tO8X5eKLLy6WWWaZln+/9dZbxUUXXVRstdVWXXqel156qTjooIOKL37xi8XOO+/c58vZm2655ZbikksuKZ588sniIx/5SDFu3Lhi+PDhPXrO999/Pz3P66+/Xpx44onFKqus0q3nueqqq4pf/epXxamnnlosvfTSRV+bMWNGev9POOGEXn/u9957Lz33ddddl9bLxhtvXBx44IHFQgstVAwUajZ/zd54441pO4rwdPnlly8+9alPpW2pp3qrZq+55ppUrz//+c+LZZddtuhrjz32WFof//3f/53WR2959913iwsuuKCYMmVKWidrrLFGqtfebKMKajZvzUZdxRddN9xwQ/qAGNvPxz72sWKHHXboledWs//x73//u/jjH/9YTJs2LdVsfI7Zb7/9io9+9KPFQKJm89XsHXfcUfzgBz9o8++bb755cfjhh3f7+dXsf5xxxhnFFVdc0ea6+vKXv1xsv/32xUCgZvMeZ2PfH58D77rrruLVV19N57Kf+cxnenwuG9Ts7B599NH0Ht5///3FAgssUKy99trF2LFjiw984APFgO8xVTuhiBOgT3/608Wee+6ZwqjPfvazxejRo4s333yzqNqiiy6aTkZ23333Lj0uPgTFa3n44YeLgSKK7atf/Wqxyy67FPPMM0/6f2xY8f/p06f36LmvvPLK4uyzzy4mT55c/PrXv+728zzyyCNpvb722mvFQPbss88Wm266afG9730v7TDjpCS2lfiQM5Co2byOOeaYYuutt049O2MbigNwbEMTJ07s8XP3Vs3GQStqNpZtoHrjjTfSB+JDDjmk+PCHP1yMGjWquPbaa4s111yzuO2224qBRM3mPcbGNvP73/+++NCHPpQCqdi29thjj2KnnXZK4WdPqNnZv2DbYIMN0onJuuuuW2y22WbF7bffngLlvviiqS+p2XxWWGGFdD7S+idOwOK4Fl9e94Sa/Y8NN9yw4bq+77770rpefPHFi4FCzeZz9dVXpy984gvVCKKiY0kcD1ZbbbXUsaGn1Ox/fPe7302ZTXQaiWPs6quvnr6EjvPaOPb2mbIiZ599dhnNXXLJJbPdfuKJJ6bbTz311D5fhgUWWKA89NBDe/w8TzzxRFrmU045pezPy1nv+9//frnwwguX991332y3v/nmm+XLL7/co+ceM2ZMucYaa5QHH3xwucwyy5Rvv/12t54ntoFYr48++mhZhTPPPDO199BDD/Xac77//vvlVlttVW666ablG2+8Mdvfnn766XIgUbP5avbmm29O2+Y3v/nN2W7/zne+U84777zlvffe2y9q9vTTT+/1GurMNnn//ff32nMee+yx6Tmvueaaltvee++9cuONNy5XXXXV8p133ikHCjWbr2Zj3//kk0/OcfvEiRPT9nXuuef26PnV7H+88MILcxxfwy677JL2j88//3w5UKjZvJ+NG9lwww3LoUOHlk899VSPnkfNtu+ZZ55J6znW90CiZvPVbKz7b3zjG+Vbb7012+377LNPOd9885X/+te/evT8avY/vvjFL87xuWXWrFnlCiusUG6wwQZlX6lsKF9bosfOkUcembrUhm9/+9vFfPPNl5K6s846K3WJX2+99Yrm5ub09wceeKA455xzin/+85+px0/0JIihgNELqF70BPjTn/6Ufv/4xz+eflqLnggxXCN+dtttt9n+9uCDDxbnnXde6umy5JJLpm88Y1mj/VrX3uiaGvMU1V7HF77whZbH9+ZytuWkk04qbr311uJ3v/tdu93qojdDfIsYQw/jG8V68c1Q/HRXDDGK5f/JT35SbLvttimxvvzyy4u999674f0jeY3uvM8991yx8sorp+6XkXT/4Q9/aOm58ZWvfKXl9Rx99NHFyJEj02uMIRLnn3/+bM8X3fhjWM+xxx5brLXWWrPdFoYMGZKGz8U32QcccECx4oordut1xhCqGCJw3HHHtXu/eG2xzcbrXHDBBWf7WxVDnXLUbNRqmDBhQnqf/va3v6XunrU6UbNdr9nLLrss/T96lNaLLrTHH3988Zvf/CbVXBU1G8P1/vKXv6SegCNGjEhzG8R+JPaPp512WrrP17/+9WLhhRdOvx911FHp29Go6Rg2fOGFF872fHfffXeq1/HjxxfrrLPObLfV12y0ETUbvZi6I5azqamp+P73v9/u/S699NL0zfl2223Xclvsp2Pdf+Mb30i9p3bcccdiIFOzfX+cje220dDP6NkTon66S83OLj6TNbL++uunY/CLL76YpooYyNRs39dsI//4xz/SZ5u99tqrWG655YruUrMdi3O86EkaU6MMBmq272s2eiDH58LWYuRTfNa8+eab07l2d6jZ2f3oRz8qFltssTlGmsVIjvj8H6Obap/7B9Xk5xEOhdqGGCcBMS9PnBTECe0mm2zSMjl1nATFCW+MdYzhFhFsRIgRJ1TRjb5+rHJMAhYrbIsttkjD9Rp1767NMfXQQw/NdvsvfvGLFHJE1/B4fHRli9viBDzmPophiK27psbvNb29nG35+9//npb/7bffbvd+EZ7FmNw4CY0Dwec///nic5/7XBoSNGvWrKInYpL2CLbihDm61G+55ZbF6aefPsf93nnnnXSgj51HHIhi6Mz8889f7Lvvvi3hY23unNjx1NZr7YNBfFiI19pabBtx+/PPP99yWzym9vh4/hhWFzvLeE9bv9edFW3E/DMdiUAhgtXo8hgnxLEDjSFCscOsf+8HstY1G+slfmIbuOeee9L6rp2Eqdnu1ezLL7/c8ASs9u94nr6u2ajTT37yk+nDVixv1GyErTGPS+xTIlSq1WwE+7Wai5CnFjY1qtnYNuL2+hP1GPpUX7PRbThOMCNQjuNAd8S4+DiedCTWdaMT3d5Y1/2Fmu3742wjsc+PD9pRN62//OoKNduxmTNnpi/6Yp8WQw0GOjWbp2bjC+fwpS99qegJNdux3/72t8UiiyySPlMMBmq272u2dVBS35kkDBs2rOguNdv5dR2faVp3vug1ZcaujzF8ZO+99063//Wvf023xRCoBRdcsPz973/fcr/osn3nnXeW88wzT3nkkUfO9rzTp09Pt//6179O/7766qvT8/3sZz+b7X7Nzc3pfvVdCqO7d9z35JNPbrnttttuK4cMGZLu36jbaUdD+fpiOdvy97//vbzgggs6HIbz4x//OLU1fPjwcr311it/8YtfpNuiO95KK61Uzpgxo+zu0IWPfOQj5UEHHdRy2x/+8Ie0/I899ljD4TJXXXXVbLfHMJnolt/RUL7DDz88ddFv7YorrkiPmTJlSrvLGu2svfba5ac+9aluDeW76KKLOmwjjB49uvzABz5QLr300uW+++5bnnHG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OsftH/rQh4qtt966X9TsjBkzittvv71Hz9EVv//974vPfvazxf3335/qqjdEMH3FFVc0/Nu6665bfPSjHy0GCjWb/zg7a9as4t577021G8eVlVdeudeeW83O7sknn0z7ggUWWCDtD+Lz3ECjZjvPcdZxtj9Qs53nODtwazbXcXZoUZG//e1vaaVtvvnmxfLLL596K911113Fyy+/XBx//PHFoYceWlQtVvInPvGJYtVVV+3S41588cXik5/8ZHHKKacUX/va14qBIEKY008/PQVTL730UjF8+PDinnvu6ZXnfuCBB4rdd989/X7aaacVBx98cLee569//Wvx5S9/OYU5TU1NxUAUJ/rHHntsccEFFxTzzTdf8eCDDxZHHnlkccIJJxQDjZrNK/YtsT3FPrN1WNLTYKq3avbqq68uvvSlL/VZuFuFp59+Ou3P44uG1iFCHCMGUjClZvOJntdHHXVUcdlll6UaXXzxxYtbbrklffkWNbbaaqv16PnV7H/85S9/KY477rhi2rRpxZZbbpn2kxGQjxs3Ln3RWMUXW71FzeblOFsNx9m+NbeczzrODv7jbGXBVM03v/nNYsyYMen3CKc+9alPpd5K0aNk++23r3RZotdQdH2bG8S3uCeddFKxxRZbFGussUavPvdvfvObYrHFFkvvYfze3ZPcwSAKd9iwYemE/b333uvyQaI/UrP5bLzxxn2yj1KzczrooINSD5rBQM1W7+abby7uvvvu4vrrry822WSTdNuzzz5bbLPNNsWOO+6YgqWFFlqo28+vZv/jnHPOKZZddtnUYzN6kNYCnugFHsfg3/3ud8VAo2bzcZytjuNs35hbzmcdZwf/cbbyYKre/PPPX/zXf/1XMXny5PQtYwRTU6ZMKeaZZ55i2223TUlupHNxor/pppu2PO6+++5Lw74irYsPgIssskjD4Rk33XRT+j16HDTqShfBWLQ7cuTIht/2R2+XSGeXXHLJYr311ksfKmOZIkUMkSLWdgTx+Hieer21nG2Jx0UXuz333DP1zmlPX/VIi3UYG+fYsWPTexaJfayX1uui5p133kk95Z577rnUOyFCsiFDhqTeW3F7iGE1ta6C0TMkCiI+8EevjL333nu254sP/nEiEG3XHlO7LcRzx/sW7YwYMaLbr/Piiy8ullhiiWLUqFHt3i/u861vfSv93t/mLuurmp06dWr6W6yb6I132223paBys802a3mcmu16zfaV7tRs/D3qKmpozTXXTHUVw5ViuGpt2GEcwMJWW22Vfo+ajpPxffbZZ7bni9qPg1ucsC+zzDKz3dabNRtzMMSxY/To0cXcTM1Wc5yNXlJx3+gpVRPHrjj2fvWrX03b90477dSt91DNznlyG58NYl9RE/uTWL/nn39+ccYZZ2Tbv/YGNVvdZ+O+ombnLmrWcXawfTY+KNNxNmswVUt5Q20C6ugKH2NSY+6fGP4Ub86KK65Y/OEPf0hv5AEHHFA88sgjxYYbbpjmiIokL7ogxjDBmgiO4t8RCMUbGMPXYgW29u9//zt1YWw9x1ScRMdJW7QX3fDffffd1A015qKIYR2XXnppul9seDNnzky/xxwrtRO73l7Otpx44olpWWI4ZGxkOURA8fzzz6cheNE7aIUVVkhDBn/xi1/Mcd9zzz23OOyww9L8YtG7Kj44xPsfqWxM9h6BRrj88stbughG4Bcf7s8666y0/uO9qBdhVryHEWjWQqN4r84777z0e0yhFoUdQyoi5b3ooovS9tVV++67bwpabrjhhmJu17pmjz766PT/6G0Sw3Lj/Yqdb7wHarZnNRv7qBjiGh9y40C01lprVVqzf/zjH9OJdRyY1llnneKpp55KJwixP46wuFazMfdN7cQhljPe/xjzHq+39TSG//jHP1LNXnXVVcXHPvaxdFuEXvU1G8sX34xtt9126SDanZ4m++23X9onx/N09mIIse+JD5jxWmPI+WChZvv+OBvBVCPvv/9++v/rr79edJeanV18OG5rXce+MsL0gRxMBTVb3Wdjx1nH2d6gZh1nB9Nn421yHWfLipx99tnxDpSXXHLJbLefdNJJ6fZf/vKX6d+bbrppudJKK5WHHHJI+e6776bb/vd//7d88cUXy+WXX77ccsstyxdeeKHl8T/5yU/KeeaZp7z55pvTv2fMmFEuvPDC5Sc+8Yny7bffTre9/PLL5T777FPON9985aGHHtry2Oeffz61ffLJJ89227LLLpuW4+mnn265/dlnny0vu+yy9PsTTzyRHnfKKafM8Tr7YjnbcuKJJ6bHv/baa2VXrLbaauVaa61V9oYddtih3G677Vr+feyxx5bDhg0rX3/99dnu95e//KUcMmRIel3vvPNOy+233nprOX369PT7qaeemtbr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",
      "text/plain": [
       "<Figure size 1200x600 with 10 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,6))\n",
    "\n",
    "for i in range(10):\n",
    "    plt.subplot(2,5,i+1)\n",
    "    plt.imshow(x_test[i].reshape(8,8), cmap='gray_r')\n",
    "    plt.title(f\" Predicted : {y_pred[i]}, Actual : {y_test[i]}\")\n",
    "    plt.axis('off')\n",
    "plt.suptitle(\"SVM classification of Handwritten Digits\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2fd713c4-5022-4818-b984-6fef641653ab",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "afb917f1-d5c9-4a04-a730-e6fdfb0eda9e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "51999ed5-928d-412d-b417-3e9aa553423e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.cluster import KMeans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "21b94d1a-b8d2-4a70-9fdc-81e80f56ac55",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "First 5 rows of the dataset:\n",
      "   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species\n",
      "0   1            5.1           3.5            1.4           0.2  Iris-setosa\n",
      "1   2            4.9           3.0            1.4           0.2  Iris-setosa\n",
      "2   3            4.7           3.2            1.3           0.2  Iris-setosa\n",
      "3   4            4.6           3.1            1.5           0.2  Iris-setosa\n",
      "4   5            5.0           3.6            1.4           0.2  Iris-setosa\n",
      "\n",
      "Dataset shape: (150, 6)\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(\"Iris.csv\")\n",
    "\n",
    "print(\"First 5 rows of the dataset:\")\n",
    "print(data.head())\n",
    "\n",
    "print(\"\\nDataset shape:\", data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "bbb86f95-bd2e-47d0-842c-cfc9dff348bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Selected features:\n",
      "   SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm\n",
      "0            5.1           3.5            1.4           0.2\n",
      "1            4.9           3.0            1.4           0.2\n",
      "2            4.7           3.2            1.3           0.2\n",
      "3            4.6           3.1            1.5           0.2\n",
      "4            5.0           3.6            1.4           0.2\n"
     ]
    }
   ],
   "source": [
    "X = data[\n",
    "    [\n",
    "        \"SepalLengthCm\",\n",
    "        \"SepalWidthCm\",\n",
    "        \"PetalLengthCm\",\n",
    "        \"PetalWidthCm\"\n",
    "    ]\n",
    "]\n",
    "\n",
    "print(\"Selected features:\")\n",
    "print(X.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "d8d6be92-e587-49e8-9a70-9de318e6bab7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n",
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n",
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n",
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n",
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n",
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n",
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WCSS values:\n",
      "[680.8244000000001, 152.36870647733906, 78.940841426146, 57.317873214285704, 46.53558205128205, 38.930963049671746, 34.36020891600984, 30.023416358198972, 28.14550248961776, 25.95154437229438]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n",
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n",
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "wcss = []\n",
    "\n",
    "for i in range(1, 11):\n",
    "    kmeans = KMeans(\n",
    "        n_clusters=i,\n",
    "        random_state=42,\n",
    "        n_init=10\n",
    "    )\n",
    "\n",
    "    kmeans.fit(X)\n",
    "    wcss.append(kmeans.inertia_)\n",
    "\n",
    "print(\"WCSS values:\")\n",
    "print(wcss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "074d95e3-aac8-4f32-a169-13390132f293",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAArgAAAHVCAYAAAAaQog2AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjExLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvctoD+AAAAAlwSFlzAAAPYQAAD2EBqD+naQAAVkdJREFUeJzt3Qd8VFXax/EnPYQ0QuiE3ntApNkFpKhYFsUuFhTX1RVf3WXdFdlVcdW1s+raWRULigpYQFZRFJUqIILU0EsIKSQhpMz7eU4yQyaFTNqUe3/f/cxO5s7JzC0J/nPmOecEORwOhwAAAAAWEezrHQAAAADqEgEXAAAAlkLABQAAgKUQcAEAAGApBFwAAABYCgEXAAAAlkLABQAAgKUQcAEAAGApBFwAqMTPP/8s3bp1k6+//tq1benSpWbbTz/9xHmrgDfPz4EDB8x7vfXWW1wLAG4IuABsZfv27SYUVXa78847XW1zc3Nl06ZNcvToUdc2/Vq35eTkiL8f33PPPVdhm//85z+uNps3b672e2h41e/VMFuWN89Pfn6+ea8jR47U+3sBCCyhvt4BAPCmvLw8E4puvfVWtzDrFBsba4njCwsLk5kzZ8rtt99ero0G323btpmAqO2rS8Nr2eAPAP6EgAvAlho3bmx6Ia1q9OjR8sknn8gPP/wggwcPdm1fuXKlrFu3Ti666CL56KOPfLqPAFBfKFEAgBr6+OOPZfjw4ZKcnCyTJk2SXbt2lWuTnZ0tM2bMkDPPPFP69Okj559/vrz99tuu57UXtW/fvvLUU0+5fZ+21wCekpLi2vbNN9+YbStWrKhy34YMGSJdu3aV119/3W37a6+9Jp07d5Zhw4ZV+H2FhYXy6quvytixY83+nnXWWfLkk0+a/VRz586Va6+91nx98803u0odZs2aVe61Fi5cKKNGjZJ+/frJDTfcIHv27Kn2+XFKS0uTu+++WwYMGCBnnHGGvPTSS1WeAwD2RQ8uANTAnDlzJCsrS6ZNmyYZGRnyl7/8RQYNGmTCZ8uWLV0f5WsY0+D7+OOPmyD4xRdfmIC4bNkyefbZZ00pQUJCgrz33nvyxz/+0Xyf1sVqmA0JCTEhUYOkWrBggezcuVN69erl0T5ef/318s9//tOE58jISFOOMHv2bBMUKwu3F198sXz77bcyffp0E5K3bt0q99xzj3z55Zcyb948Oeecc+TBBx+U6667zhz7aaedZr63WbNmbq81f/58Uxt73333SWZmpimVGDNmjKxZs0aCgoI8Pj/OEHz66aeb19F2bdu2Nedfe6IBoEIOALCRX3/91aH/9DVu3NjRtWvXcrc333zT1XbZsmWm7bx581zbPvvsM7PtnHPOcXvdffv2OSIjIx033HCDa9tDDz1k2i5ZssSt7bRp08x2fX314IMPOkJCQhzp6enm8cyZM83+jRs3zjF+/HjX9yUnJztGjBjh0fHNmDHDsXv3bkdwcLBj9uzZ5rl3333XPN61a5fjscceM+3WrVvn+t5///vfZtvnn3/u9po//vij2f7++++bx1999ZV5rOeiLOf5GTVqlNv2uXPnmu2LFi2q0fnRx6tXr3Zrp+datz/77LMnPScA7IcSBQC2NH78eFODWvamH8174oorrnB73Lx5cxk5cqTpuXTSHtf27dubXsrSJk6caO6dbUeMGGF6TxcvXmweL1q0yJQ+nHfeeWZbUVGRpKammt5PbeupVq1amfbOMgUtT9DXbd26dYXttRdZv0fft7RTTz3V9Ep/9tlnHr/35ZdfXu41lA5Oq+750XY9e/Y0pQ6lXX311R7vDwB7oUQBgC3VdpBZUlJShdt0YJeGVS0v2Lt3r/k4vaJ2wcHB5nl1yimnSKNGjUw5wrhx4+Srr76SJ554wtSl3nbbbWZgmJYKOByOagVcZ5nCVVddJcuXLzfB+c0336y0rdb7almBswRC3895O3z4sOzfv7/G50fLMJQGdSdPz8++ffsqvFYVXQMAUARcAKgBHfRU0baoqCgTblVMTEyFc7TqNu2V1eeVhjmtbdWAq3PMak2vBlkNcNrDqdt1Wq8mTZqYAWnVobMl6NRnl112mXk/fVwZfb5hw4by/vvvV/h8dHS0x+/rPAdlaVgu/X6enB9938rONwBUhBIFAKiB0qubKZ1lQBc+GDp0qGubfr1x48ZyswfogC1VeiYDDbS6SMPzzz9vZj9w9k7qdg24+j1aXuAcoOUpHVym5QI7duyQCRMmmMeV0ZCtA9zi4+MrXATDWdrgfI2CggKpDU/Pj7Zbv3696UUuTXu6AaAiBFwAqAGtJXXOI6vhVmcm2L17t0ydOtXVRrfpLAk6RVZ6errZpkHt3nvvNVNi6YwFTs7SA112Vmt5nfRrndVAZ0+obnmCk86k8Ouvv5r7k/nTn/4kiYmJcumll8pvv/3m1lOqU4V9/vnn5rH2Kmuvs9YE14an5+euu+4yPbo6FZvOqKC+//57M9MEAFSEgAvAll544YUKeykvvPBCj75fg+yHH35opseKi4szA7Q0nGovqJPON6u9kRoQtbxAB2r1799fBg4caOphNdw5dejQwdw0yJUOsvp6GiaV9uDWhO6fHltVq7TpQDldGEJ7arUUQvdZj0+/V3tZdb5fpdt0WrS///3vpoa2snlwq+Lp+dHX1wFnGqi1VrlFixZmGjOdPxcAKhKkUylU+AwAWNDx48dNPWtlIiIiTA+lOnbsmPloX8sFtDZVaQ+iztvapk0bU2+r9bJaM6ptKqs7VRritJdSQ6R+X0W0B1iXv+3YsaNb+NUBZjpwrUuXLh4fX9OmTV0Duyrbn4MHD5pQHR4eXuHr6CAvrX/VXt2K5Obmmjbag62hV8Nn2fPjpP+p0V5vfa2KXs+T86OvoeeoQYMG5jW0RGLLli3me7SsAgCcCLgAAACwFEoUAAAAYCkEXAAAAFgKARcAAACWQsAFAACApRBwAQAAYCks1Sti5p3UqW50WcjqrhIEAACA+qdTBWZlZZk5s53zg1eGgCtiwq1zWUwAAAD4L51r27l0eGUIuCKm59Z5wqpa6acu6KToura8LsFZejJ3O+DY7XfdueZcczvh552fdzvJ93KeyczMNB2Sztx2MgRcXe2ipCxBw623Aq6u1KPvZaegozh2+113rjnX3E74eefn3U7yfZRnPCkn9dkgs5dfflkiIyMrvO3cudOtfOCyyy6Txo0bS6tWrWTKlCmSl5fn9lqetAEAAIA9+KwHd+LEiXL11Ve7bbvkkkvkwIEDZg1zpWuvjx071gTXH3/80az3fumll5q1zl988UWP2wAAAMA+fBZwQ0JCzM1p3759po7jueeec23Tx2vWrJGtW7dKhw4dzLaHH35Yrr/+ennooYckMTHRozYAAACwD7+ZB/f111+XiIgIufLKK13bli5dKm3btnUFV3XuueeaXtsffvjB4zYAAACwD78ZZPbqq6/K5Zdf7jbIS3t1mzZt6tauSZMmprh4//79HrcpS+tzS9fo6qg8Z7G03uqb8z288V7+hmO333XnmnPN7YSfd37e7STfy3mmOu/jFwF3yZIlsmXLFpk1a1a558pO5OscOaeT/VanTWkzZsyQ6dOnl9uu5Q46GtBbFi1aJHbFsdsP19x+uOb2ZNfrbtfj9uax5+TkBFbAfeWVV6Rnz54yZMgQt+3NmjUz4be01NRUE1z1OU/blDV16lQz00LZedV0HjdvTROmPwwjRoyw1XRRimO333XnmnPN7YSfd37e7STfy3nG+Yl7QATcjIwMmTNnjulVLUsD7yOPPGIWYHCuNPbVV1+ZHttTTz3V4zZlaa2v3srSi+PN4OHt9/MnHLv9rjvXnGtuJ/y88/NuJ2FeyjPVeQ+fDzKbPXu2FBUVyTXXXFPuudGjR0u3bt3k9ttvN72ymzdvlmnTpskVV1whzZs397gNAAAA7CPYH8oTdN7ahISECpP6ggULTM1Fy5YtpV+/fjJ06FB54YUXqtUGAAAA9uHzEgWd5is0tPLd0Om/tL5De3nLDiarTht/UVjkkB+3p8nK1CBpvD1NhnRqKiHBVS85BwAAgAAJuBXVwlbEk+Dq7+H28/X7ZPq8DbIv45gudSGzNq+QFnGRMu2CHjKqVwtf7x4AAIAl+HcitBANt5PfXFUSbk/Yn3HMbNfnAQAAUHsEXC+VJWjPbUWz8jq36fPaDgAAALVDwPWCn7anleu5LU1jrT6v7QAAAFA7BFwvOJh1rE7bAQAAoHIEXC9oGhNZp+0AAABQOQKuF5zaPsHMllDZZGC6XZ/XdgAAAKgdAq4X6Dy3OhWYqizk6vPMhwsAAFB7BFwv0Xlun7+6vzSPcy9DiGsQarYzDy4AAEDdIOB6kYbYpX86R9684RTpk1Bktp3dtSnhFgAAoA4RcL1MyxAGtU+QoU2L57xdvSvd27sAAABgaQRcH2kbUxxwUw7nyOGjeb7aDQAAAMsh4PpIVKhIxyYNzderd9KLCwAAUFcIuD7ULynO3K/aecSXuwEAAGApBFwfSk6KN/cEXAAAgLpDwPWh5JIe3LW7M6SgsHhWBQAAANQOAdeHOjWJlpiIUMk5XiibDmT5clcAAAAsg4Dry5MfHCR9XWUKDDQDAACoCwRcH+vfpjjgrk5hoBkAAEBdIOD6WHLbRuaeBR8AAADqBgHXx/onFQfc7anZkpZ93Ne7AwAAEPAIuD4WFxVWasEHyhQAAABqi4DrB5LbFPfiMh8uAABA7RFw/UB/Z8BNYSYFAACA2iLg+oH+bYtnUvh5d7oUFjl8vTsAAAABjYDrBzo3jZFo54IP+1nwAQAAoDYIuH4gxCz4ULxsL3W4AAAAtUPA9bc6XGZSAAAAqBUCrp8F3DUs2QsAAFArBFw/0S+peKDZttRsOcKCDwAAADVGwPUTjRqGS4fEkgUfdrHgAwAAQE0RcP1xwQfmwwUAAKgxAq4fzodLDy4AAEDNEXD9dKAZCz4AAADUDAHXj3RpFiMNw0Mk+3ih/HaABR8AAABqgoDrdws+FJcpMB8uAABAzRBw/bRMYTXz4QIAANQIAddPB5rRgwsAAFAzBFw/k5xU3IO77VC2pOcc9/XuAAAABBwCrh8u+NDeueADZQoAAADVRsD1Q8ltKFMAAACoKQKuH2KgGQAAQM2Fio8dPXpU5s6dK9u3b5fk5GS54IILyrX5/vvvZfHixRIZGSkXXXSRdO7cuUZtAm7Bh13FCz7o9GEAAAAIgB7cdevWSbdu3eTFF1+UoqIieeutt+T66693a/Pggw/KeeedJwcOHJA1a9ZI7969ZcGCBdVuE0i6NIuWqPAQOZpXIJsPsuADAABAQPTgFhQUyO9+9zsZNmyYvPvuu67tGzdudH29bds2eeCBB2T27Nkyfvx4sy0xMVEmT54sO3bskODgYI/aBJrQkGDp2zpelm07LKtS0qVb81hf7xIAAEDA8Fn6++KLL+S3336T6dOnu23XHl2nTz75RBo2bCgXX3yxa9vEiRNl165dsnz5co/bBPJ8uKt3HvH1rgAAAAQUn/Xg/vTTT9KkSRNp3LixPP3005KTk2NqcEeNGuVqowG4bdu2Ehp6Yjc7derkem7QoEEetSkrLy/P3JwyMzPNfX5+vrnVN+d7nOy9+rQq7rVdmXLEK/vkLZ4cu1XZ9djtetx2Pna7Hrfi2O133bnm4rVrXp338VnAzcjIkKCgIDnjjDNkxIgRZnDYddddJ0OHDpUPP/zQPJednS2xse4fz0dHR0tISIh5TnnSpqwZM2aU6zlWCxculKioKPGWRYsWVfrcUXMNQ2Vbara8//Gn0jBMLOVkx251dj12ux63nY/drsetOHb74ZrXP+0M9fuAqyH04MGD8sorr8j5559vtl1zzTXSp08fEzR10Ji20SBcWlZWlhQWFprnnK9TVZuypk6dKlOmTHHrwU1KSpKRI0eWC8v19ReI/iJosA8Lqzy5/mfbUklJy5Gm3QfKmV2aiBV4euxWZNdjt+tx2/nY7XrcimO333Xnmi/y2jV3fuLu1wG3R48e5v7UU091bdPZDxo0aCBbtmwxAbd79+4ya9YsOX78uISHh5s2mzZtMvf6nPO+qjZlRUREmFtZenG8+UtZ1fsNaNvIBNy1e4/K8J4txUq8fa79iV2P3a7Hbedjt+txK47dftedax7mlXPs94PMxowZIzExMfL111+71eXm5uaaoKvGjRtnamV1hgQnnVKsQ4cO0r9/f4/bBKrktsXz4TLQDAAAQPy/Bzc+Pt4E0Ztuuknmz59vanDfe+89M72X1uUqLRt47LHHzLYvv/xS0tLSZMmSJTJv3jxTo+tpm0CVnFQ8k8KanelSVOSQYBZ8AAAA8O+VzK644goZPHiwfP7552a+2ltvvbVcr+udd94p55xzjunp1bKCl156SVq2bFntNoGoW/MYs+BDllnw4ah0bR7j610CAADwez5fqrd9+/am9/VktGTBWbZQmzaBuOBDn9Zx8sO2NFOmQMAFAACoWuAt82Uz/dsU1+GuYsEHAAAAjxBwAybgpvt6VwAAAAICAdfP9WtTPNBsy8GjkpFjr9VhAAAAaoKA6+cSoyOkbePi1dVW7zri690BAADwewTcACpTWE2ZAgAAQJUIuAGgf0mZAgPNAAAAqkbADQDJJT24a3YVL/gAAACAyhFwA2TBhwZhIZJ1rEC2HDrq690BAADwawTcAFrwQemCDwAAAKgcATdA9G9bMh9uCvPhAgAAnAwBN0AkJzHQDAAAwBME3ADrwd2sCz7ksuADAABAZQi4AbTgQ5uE4gUfft5FmQIAAEBlCLgBhPlwAQAAqkbADcD5cFexohkAAEClCLgBuWTvERZ8AAAAqAQBN4B0axEjkWHBZsGHbaks+AAAAFARAm4ACTMLPpRMF8Z8uAAAABUi4AZomcIqVjQDAACoEAE3wCS3YcEHAACAkyHgBmgPri74kHmMBR8AAADKIuAGmCYxEZKU0EAcDhZ8AAAAqAgBN5DrcBloBgAAUA4BNwAlJ1GHCwAAUBkCbgDq35YFHwAAACpDwA1A3VvEmgUfMs2CD9m+3h0AAAC/QsAN1AUfWlGmAAAAUBECboDPh7uaBR8AAADcEHADVDIzKQAAAFSIgBug+rct7sH97WCWZLHgAwAAgAsBN0A1jYmU1o2cCz5k+Hp3AAAA/AYB1wplCtThAgAAuBBwA1j/koFmBFwAAIATCLgWWLJ39c50cWitAgAAAAi4gb7gQ0RosGTk5rPgAwAAQAl6cANYeGiw9GkdZ75elXLE17sDAADgFwi4lhlolu7rXQEAAPALBFyLDDRjRTMAAIBiBFyLDDTbdCBLjuYV+Hp3AAAAfI6AG+CaxkZKq3jngg+UKQAAABBwLSDZOR8uA80AAAAk1JfnIDU1tdy2mJgYiYiIKLc9NzdXQkNDJSwsrNLX86SNVcsU5q/dx4IPAAAAvixROHr0qDRp0kQ6deok3bp1c93mzp3r1m7jxo0ydOhQiYuLk4YNG8pll10mGRkZ1W5jZf3bliz4sIsFHwAAAHxeovD555+bnlznbcKECa7n8vLyZOzYsdKmTRtJS0uTHTt2yIYNG+Smm26qVhur69Ei1syJm56TL9tTs329OwAAAPYtUVBFRUWSnZ1tel7L+vTTT2X79u3y7bffSnR0tLk98MADpod279690rJlS4/aWJ2G296t4mRlyhEzH26HJtG+3iUAAAD79uCeffbZ0rhxY2nevLlMmzZNjh8/7nruhx9+kA4dOriF1DPPPFMcDof89NNPHrex03y4q3ayohkAALA3n/XgBgcHy1/+8he56667JCEhQf73v//J5ZdfbnpzH3/8cdPm0KFDpk63NA3D+r36nKdtytKyBr05ZWZmmvv8/Hxzq2/O96jL9+rTKtY1k4I3jsGfjj1Q2PXY7Xrcdj52ux634tjtd9255uK1a16d9wlyaFenn9Bgq+UFWVlZEhQUJDfeeKOsW7fOrSe2oKDAzJLw8ssvm+c9aVOWvsf06dPLbX/77bclKipKAlHGcZH7V4ZKkDjkkVMLJTLE13sEAABQd3JycuTKK680EwnExhZ37PltDW5pnTt3Nj24Bw4cMCULWnbw5ZdfurXR51SLFi3MvSdtypo6dapMmTLFrQc3KSlJRo4cWeUJq6u/QBYtWiQjRoyo0ynNnt/yjezLOCYteg6SIR0aiz+qr2MPBHY9drset52P3a7HrTh2+113rvkir11z5yfunvBZwNWOY+2lLW358uVmkJiWGKgzzjhDHnzwQdm8ebMJv2rhwoXmJA4ePNjjNmXpPLsVzbWr3+PNX8q6fj+dLmzB2n2ydk+WnNG1ufgzb59rf2LXY7frcdv52O163Ipjt99155qHeeUc+33AffrppyU9PV0uvPBCiY+PN7Mh/Otf/zK9q84DOPfcc2XQoEEyceJEmTlzppkG7L777pNbb73V1O162sZOCz5owF29kyV7AQCAffks4GoAffbZZ829DgbTmRDeeOMNM72Xkw4Umzdvntx7770mCGuvq9bU3n///dVqY7eZFJwLPpTtIQcAALADnwXcyMhIueeee8ztZHSGhNdee63WbeygZ8s4MyduWvZx2XE4R9onlp9bGAAAwOp8Pg8u6o6G214tT0wXBgAAYEcEXAvW4SoWfAAAAHZFwLUYnUlBMdAMAADYFQHXoj24G/dnSnZega93BwAAwOsIuBbTPC5SWsRFSpFD5OfdTBcGAADsh4Br4V5cyhQAAIAdEXAtKNk5H+5OZlIAAAD2Q8C18ECzVTuLF3wAAACwEwKuBfVsGSvhIcULPqQczvH17gAAAHgVAdeCIkJDpGerkgUfKFMAAAA2Q8C1KAaaAQAAuyLgWhQrmgEAALsi4Fp8JoWN+7Mk5zgLPgAAAPsg4FpUy/gG0jw2UgqLHPLzrgxf7w4AAIDXEHAtrH/bkvlwdzEfLgAAsA8Crh3qcFNYshcAANgHAdfCkl1L9h5hwQcAAGAbBFyLL/gQFhIkh7OPy840FnwAAAD2QMC1sMiwEOnZMs58zYIPAADALgi4FseCDwAAwG4IuDaZSYEeXAAAYBcEXJsMNPt1Hws+AAAAeyDgWlzLuEhpFhthFnxYu5sFHwAAgPURcC0uKCiIOlwAAGArBFw7LfiwkxXNAACA9RFwbSC5TcmSvSz4AAAAbICAawO9WsWZBR9Sjx6XXWm5vt4dAACAekXAtcmCDz1KFnxYvYsyBQAAYG0EXJvoX1KmsCqFgAsAAKyNgGuz+XBX7Uz39a4AAADUKwKuzXpwf92XKbnHC329OwAAAPWGgGsTreIbSNOYCCkocsi6PSz4AAAArIuAa8MFH5gPFwAAWBkB10b6t2WgGQAAsD4Crk0HmjkcDl/vDgAAQL0g4NpI71ZxEhqsCz7kye4jLPgAAACsiYBrswUferaMNV9ThwsAAKyKgGszzjKF1cyHCwAALIqAazPJzhXNdrKiGQAAsCYCrs04pwrbsDdTjuWz4AMAALAeAq7NtG7UQJqw4AMAALAwAq4tF3xgPlwAAGBdoeIHdE7We++9V/bt2yfPPvusNGpU/DG603vvvSeLFy+WyMhIGT9+vJx22mnlXsOTNjgx0OyLXw5QhwsAACzJL3pwH3vsMXnnnXfkrbfekuzsbLfnfv/738sf/vAHad++vQmv55xzjsyaNavabXDCiSV7WfABAABYj897cH/66SeZOXOmCblXXHGF23O//PKL/Pvf/5aFCxfKiBEjzDYNsHfffbdpGxYW5lEbuOvTunjBh0NZebInPVdaN4riFAEAAMvwaQ9uRkaGCaH/+c9/JDExsdzzCxYskISEBDn33HNd2yZMmCCpqanyww8/eNwG5Rd86OFa8CGd0wMAACzFpz24kyZNkgsuuEDOO+88+fLLL8s9v23bNklKSpLg4BM5vF27dq7nTj/9dI/alJWXl2duTpmZmeY+Pz/f3Oqb8z288V6V6ds6TtbuzpAV2w/L6B5NvPa+/nDsvmLXY7frcdv52O163Ipjt99155qL1655dd7HZwH3pZdekg0bNsgbb7xRaZvc3FyJjo5229agQQMJCQkxz3napqwZM2bI9OnTy23XMoeoKO99XL9o0SLxmdQgEQmRr9elyICgbV5/e58eu4/Z9djtetx2Pna7Hrfi2O2Ha17/cnJy/D/gPvLII9KsWTO56aabzOP9+/eb+zvuuEMuueQSufrqqyUuLk6OHDlSrqyhsLDQPKc8aVPW1KlTZcqUKW49uNoLPHLkSImNLf7ovr7/AtFfBK0Z9lWNcO8jOfLfJ5bK3txgOWfEcFO24A3+cOy+Ytdjt+tx2/nY7XrcimO333Xnmnvvmjs/cffrgPuvf/1Ljh496nqsg8V0mq+zzz5bunfvbrb17t1bXn75ZZPYnT2r69atcz3naZuyIiIizK0svTje/KX09vuV1r5JrCRGR0jq0TzZdDBHTmmX4NX39+Wx+5pdj92ux23nY7frcSuO3X7XnWse5pVz7PeDzC666CLTS+u8OQeJXXzxxTJgwADz9bhx40xt7Ysvvuj6vqefflr69OkjvXr18rgNqljwYad7DzgAAEAgq5Me3KysLGnYsKHbQK+60LRpU1Orq2UMn3zyiSlFOHDggHz22WfVaoPKF3xYuOGArEphJgUAAGDTgKuhUafzGjhwoHl8+PBhufTSS2XJkiUSExMjTz75pNx444012hHtbf3vf/9rpvwqTacR04Ubvv/+e1NWcOaZZ5owXd02KK90D66uJqe9ugAAALYJuDojwe233+42t+zf/vY3MxOCLtSwc+dOue2228xALR2wVV3Nmzc3pQoV0cFoWrpwMp60gbs+rePNgg8Hs/Jkb8YxaRXfgFMEAADsE3C1l7ZHjx7SpEnxnKna4/fBBx/Igw8+aOazVZs3b5ZPP/1UbrnllvrbY9SZBuEh0r1FrKzbkyGrUo4QcAEAgCV4XDSriya0bdvW9XjLli1y8OBBs0iDU79+/WTPnj11v5eoN8kMNAMAAHYNuDqYS6fycvrqq6+kZcuWbqFXl8etaMld+K/+bRqZe5bsBQAAtgu4Oohr5cqVcvfdd8vrr79uShN0QYbSfvrpJxkyZEh97CfqOeBu2Jshx/ILOc8AAMA+AVdnN3jrrbdk9uzZMnHiRGnXrp1MmzbNLdxqXa5zhgUEhqSEBpIYHS75hQ75ZW+Gr3cHAACg1qo1ce0FF1wge/fuNTMqfPPNN27lCLo4w9KlS2u/R/AqnRqsX1JJmQLz4QIAAAuo0coMkZGR5RZ60KAUGuqzlX9RC/3bsqIZAACwacDVhR6WL1/ueqwLPZx11lkSGxsr8fHx8sorr9THPsJLdbird7KiGQAAsFHAdS70oLW3FS30oIs86G3Xrl31ta+oJ31ax0lIcJDszzwme9NzOc8AACCgsdADJCo8VLq3iJH1ezLNsr0tWdEMAAAEMBZ6gJHMQDMAAGARLPQAt4Fmq3cd4YwAAICAxkIPcBto9sueTMkrYMEHAAAQuFjoAUabhChp3DBcjhcWmVpcAACAQMVCDzB0HuPkNiVlCjspUwAAADYJuLoUb1FRUbmFHswLBQebGwJXckmZgs6kAAAAEKiqlUjPP/98WbhwYYXPrV69WgYPHlxX+wUfYMEHAABgq4C7ceNG2blzp4waNarC5/v3729WNFuyZEld7h+8qG9SnAQHiezLOCb7MljwAQAAWDzgag9t3759T9pGn9d2CNwFH7o1jzVfr0ph2V4AAGDxgJuamiqNGhXXaFZGn9d2CPz5cKnDBQAAlg+4rVu3ljVr1py0jT6v7WCFOlwGmgEAAIsH3LPPPtsE2A8//LDC57X2dsGCBTJ69Oi63D/4KODqXLgs+AAAAAJRqKcN4+Pj5aGHHpJLL71Ufve735nBZq1atZL9+/fL119/Lf/973/lvvvuk7Zt29bvHqNetW0cJQkNwyUt+7j8sjfTFXgBAAAsF3DVHXfcYepsp02bJnPmzHFtb9GihTzzzDPy+9//vj72Ed5e8CEpXhZvPCirUo4QcAEAgLUDrrrmmmvMLSUlxQwo057dDh06mGAEa+jftpEJuKt3MZMCAACwcMD97bffTIjt3LmzeaylCJQjWJNryd4UBpoBAAALDzL75JNPpEuXLpKUlCTXXnutvP7662bhB1hP39bxZsGHvRnHZH/GMV/vDgAAQP0E3Ntvv10WL14s119/vWzbtk1uueUW04PbsWNHufnmm+Xtt982A84Q+BpGhEpX54IPTBcGAACsGnAjIyPlnHPOkX/84x+ydOlSOXLkiCxcuFAuv/xyWb9+vVx33XVmsJk+j8DX31mmQMAFAABWH2TmFBUVZQKvDjLTW2hoqAm+2dnZdbuH8AmdHuytH3fKqp0MNAMAABYOuEVFRWaxh6+++srMffvNN99IRESEnHHGGXLZZZfJzJkzpXfv3vW3t/DqTApq3Z4MOV5QJOGhHnf2AwAABEbAfeedd2Ty5MmmVEED7ZgxY+Sf//yn9OjRo373ED7RrnGUNIoKkyM5+fLL3gxJZsEHAAAQIDzulktLS5OMjAxp3769GVimN6YJs/iCDyWhljIFAABgyYB72223ye7du81qZQcPHjSPdVWzoUOHyl/+8hcz4Iz6W2thoBkAAAhE1SqsbNmypVx11VXy8ssvy5YtW2Tr1q1y6623yr59+2TSpEkm8D7++OP1t7fw+kAztZqBZgAAIIAE1/Zj7NK3/Px8OXDgQN3tHXyqT1Lxgg970nPlQCYLPgAAAAvOorB3714ze4JzFgXtxQ0PD5dTTz1VrrnmGjNt2JAhQ+pvb+FV0RGh0qVZjGzcnyWrUo7I6N4tuAIAAMA6Afff//63qb/V+W4HDBgg48ePl7PPPluGDRtm5sSFdacL04C7elc6ARcAAFgr4GqoXbBggZx++ukSExNTv3sFv6rDfVsXfEg54utdAQAAqNuAO2jQIE+bwkKSS5bsXcuCDwAAIECwPBVOqkNiQ4mPCjOrmW3Yl8nZAgAAfs/nAbewsNAMVktNTT3pEsGbN2+WXbt21aoNarjgQ1JxL+7qnZQpAAAA/+ezgKuLQvzpT3+SJk2ayNixY80KaVoG8euvv7q1W7ZsmXTo0MEMZuvevbtZWEJnc6huG9R+PlxWNAMAAIHAZwFXF4fQpX737NkjmzZtMqujJSQkmOnGnI4ePSoXX3yxXHjhhWZ+3UOHDpkexeuuu65abVD7mRQUA80AAEAg8FnA7dSpk1nut0GDBuax3g8fPly2b9/uavPJJ5+Y0oVp06aZ0KptdFngL7/80tXOkzaonT6t4ySoZMGHgyz4AAAA/JzPa3B1ud9Vq1bJO++8I08//bTcd999rudWrlwpHTt2lMaNG7u2OReS0O/xtA1qJyYyTLo2K54ajjIFAABgqZXM6sNjjz0m3377rezYsUNOO+00mTBhguu5w4cPuwVX1ahRIwkODnYNSvOkTVl5eXnm5pSZWTw7gC41rLf65nwPb7xXXenbOs4s+LBix2E5t6v7+bb6sdcVux67XY/bzsdu1+NWHLv9rjvXXLx2zavzPj4PuC+88IKrllbrZnW53/Xr15sV08LCwtyCqPPgdMYEfU550qasGTNmyPTp08ttX7hwoVdXZVu0aJEEipAjQfr/8r+ft0vvwi22Ova6Ztdjt+tx2/nY7XrcimO3H655/cvJyQmcgOsUHR0t99xzjykv0EFnPXv2lKSkJLN6WmnO2RH0Oed9VW3Kmjp1qkyZMsWtB1fbjhw5UmJjY6W+aQDXX4QRI0ZUGsL9TddD2fL2M9/JntwQGT5yhISHBtvm2OuKXY/drsdt52O363Erjt1+151rvshr19z5ibtfB1ztdY2IiHDbtnPnTnMfFxdn7rU3VwePrVmzRvr162e2zZ8/3wwkc9bZetKmLH3fsu+t9OJ485fS2+9XG12ax0lcgzDJyM2XLam50rdkblw7HHtds+ux2/W47Xzsdj1uxbHb77pzzcO8co79PuA+++yzZmEGnQM3MTHRBFQtG9Bpwlq3bm3aaE3u6NGj5aqrrpJHH31U0tLSzCC0P//5z6bH19M2qL3g4CCzbO/Xmw6ZBR9qG3ABAADqi88C7t133y2zZ8+WN954Q/bv329C7TPPPCPjx493a/f+++/LI488Ig8//LDpddUQO2nSpGq3Qd0s+KABV2dSuH4YZxQAAPgnnwVcnbP2yiuvNLeTadiwofzjH/8wt9q0Qe1pD65axZK9AADAj/l8HlwEjn5J8WbBh91HcuVg1jFf7w4AAECFCLio1oIPXZoWL/iwemc6Zw4AAPglAi6qpX9byhQAAIB/I+CiWpKTGpn71Sn04AIAAP9EwEWNenDX7kmX/MIizh4AAPA7BFxUS4fEaImNDJVj+UWycV8WZw8AAPgdAi5qsOBDcZkC04UBAAB/RMBFjRZ8UARcAADgjwi4qDYWfAAAAP6MgItq69emeMGHXWm5cigrjzMIAAD8CgEX1RYbGSadm0abr1ezbC8AAPAzBFzUsg6X+XABAIB/IeCiRqjDBQAA/oqAi1r14K7dzYIPAADAvxBwUSMdm5xY8GHTfhZ8AAAA/oOAi5r94AQHST/mwwUAAH6IgIsaS06KN/erUo5wFgEAgN8g4KLG+rdlJgUAAOB/CLiosX4lPbg703Ik9SgLPgAAAP9AwEWNxTUoveAD8+ECAAD/QMBFrTAfLgAA8DcEXNTNimYMNAMAAH6CgIs6GWi2dneGFBQWcTYBAIDPEXBRK52aREtMZKjk5hfKRhZ8AAAAfoCAi9ov+FAym8LqncyHCwAAfI+Ai1pLdq1oxkwKAADA9wi4qLX+bejBBQAA/oOAi1pLTiruwd1xOEcOs+ADAADwMQIuai0uKkw6seADAADwEwRc1InkkoFmqxhoBgAAfIyAizqdD5eACwAAfI2Aizpd0YwFHwAAgK8RcFEnOjeNlpiIUMk5XiibDmRxVgEAgM8QcFE3P0jBQdLXVYfLfLgAAMB3CLio+/lwU1jRDAAA+A4BF3UmuWSg2epd9OACAADfIeCizqcK256aLWnZxzmzAADAJwi4qDPxUeHSoUlD8/Vq5sMFAAA+QsBFvUwXxny4AADAVwi4qJeAu5qZFAAAgI8QcFGn+rctrsP9eVe6FBY5OLsAAMDrCLioU52bxkh0RKhk64IP+1nwAQAA2Czg/vDDD/KHP/xBxo4dK7feequsWLGiXJvjx4/LE088Ydpceuml8t5779WoDbwjxCz4EGe+pg4XAADYKuC++OKLMmXKFOnatavcdtttEh8fL4MGDZKPPvrIrd1VV10lM2fOlCuuuELOOussuf76602YrW4beA8DzQAAgC+F+uqNL7/8crnllltcj7X3ddeuXfL444/LRRddZLYtX75c5syZY3p6Nfw6e2sfeOABmTx5sjRo0MCjNvBNwF3DQDMAAGCnHlztsa1oW05OjuvxwoULpVmzZq7gqjT8ZmVlybJlyzxuA+/qV7Lgw7bUbDnCgg8AAMAuPbhl7dmzR95++2258847XdtSUlKkZcuWbu1at27tes7TNmXl5eWZm1NmZqa5z8/PN7f65nwPb7yXL0SHB0n7xlGy/XCOLN+eKmd3bWKbYz8Zux67XY/bzsdu1+NWHLv9rjvXXLx2zavzPn4RcLW3ddy4caYed+rUqa7tWmpQtsQgPDxcgoODzXOetilrxowZMn369HLbtTc4KipKvGXRokViVU2Cg2W7BMucr1ZK7tYiWx17Vex67HY9bjsfu12PW3Hs9sM1r3+lP+X3+4B79OhRGT16tBQVFclnn30mERERrucaNWokhw8fdmufnp5u2upznrYpS0O0DnAr3YOblJQkI0eOlNjYWPHGXyD6izBixAgJCwsTK8pYvkt++uRXORqRKGPGnGKrY6+MXY/drsdt52O363Erjt1+151rvshr19z5ibvfB1wNt2PGjJHs7GxZvHhxuUCanJxsZkfQwOqs2dVBZapfv34etylLQ3TpIO2kF8ebv5Tefj9vGtg+0dz/vDtDgkNCzfRhdjn2qtj12O163HY+drset+LY7XfdueZhXjnHfj/ITLuZdeYEDbkabhMSEsq10bKFmJgYefTRR83jgoIC8/WwYcOkS5cuHreB93VpFiMNw0PMgg+/HWDBBwAA4D0+68HVEPrNN99I37595ZJLLnFtj46Olvnz55uv4+Li5N133zXz277//vumVrdx48au5z1tA18t+BAv3289bBZ86N6i/ks/AAAAfBpwr732WrMoQ1mhoe67NHz4cNm9e7esXbvWlBX07t1bgoKCqt0GvpkPVwPu6p3pctWgtlwCAABg7YDboUMHc/OEhtaBAwfWug28q3/b4ppoluwFAADe5LMaXFhfv6TiQYPbDmVLek7FU7YBAADUNQIu6k1Cw3Bpn9jQfK1lCgAAAN5AwEW9Sm5TXKaweucRzjQAAPAKAi7qfaCZWkUPLgAA8BICLrwScNfsSpfCIgdnGwAA1DsCLupVl2bREhUeIkfzCmTzQRZ8AAAA9Y+Ai3oVGhIsfVuXTBeWwkAzAABQ/wi48Np8uAw0AwAA3kDAhRcHmjGTAgAAqH8EXNS7fknFPbhbzYIP+ZxxAABQrwi4qHeNoyOkXeMo8/XPu6nDBQAA9YuACy9PF5bBGQcAAPWKgAuvSG5bHHBXE3ABAEA9I+DCK5JL6nBX7jwiKw4FyY/b01j4AQAA1IvQ+nlZwF3K4WwJEpFj+UXy3y0h8t8tK6RFXKRMu6CHjOrVgtMFAADqDD24qHefr98nt7+9Wsou1Ls/45hMfnOVeR4AAKCuEHBRrwqLHDJ93oZy4VY5t+nz2g4AAKAuEHBRr37anib7Mo5V+rzGWn1e2wEAANQFAi7q1cGsY3XaDgAAoCoEXNSrpjGRddoOAACgKgRc1KtT2yeY2RKCTvZDGCRyKOuYOBzU4QIAgNoj4KJehQQHmanAVGUhV8eX3fHOGrnh9eWyKy2HKwIAAGqFgIt6p/PcPn91f2ke516GoD27z17RT+48t7OEhwTLV5sOyYgnl8iLS7ZKfmERVwYAANQICz3AayF3RI/msmzLQVn47Y8y8vRBMqRTU9PDqy7o21Lum7vOrHA247ON8tGavfLwxb0kuU3xEr8AAACeogcXXqNhdlD7BBmQ6DD3znCrOjWNlncmDZZHf9dH4qPC5Nd9mXLJ89/L/R+vl8xj+VwlAADgMQIu/EZQUJBcdkqSLJ5yplyS3Ep0zNmsZSky4okl8tm6fQxCAwAAHiHgwu80jo6QJy7vJ2/dNEjaNY6SA5l5MvmtVXLzrBWyJz3X17sHAAD8HAEXfmtYp0T5/I9nyB/O6SRhIUHy5a8HTW/uy99ukwIGoQEAgEoQcOHXIsNC5O6RXeXTO06Xge0aSc7xQnlwwa8ybuZ3snZ3uq93DwAA+CECLgJC52Yx8u6kIfLIJb0lNjJUftmbKRfN/E6mz/tFjuYV+Hr3AACAHyHgImAEBwfJhFPbyOK7z5Jx/VqaBSJe+26HKVtY+Mt+X+8eAADwEwRcBJwmMRHy9IRkmXXDqdImIUr2ZRyTSf9dKZNmrZB9GQxCAwDA7gi4CFhndGkiX/zxDLntrI4SGhwkCzcckOH/WiKvfbddCrV7FwAA2BIBFwGtQXiI3Duqm8y/4zTp3yZeso8XyvR5G+Tif38n6/dk+Hr3AACADxBwYQndmsfKnFuHyoMX9ZKYyFBZuztDLnxuqTw4f4NkMwgNAABbIeDCUoPQrh7c1qyENrZPCzMI7eWl22Xkk9/I4l8P+Hr3AACAlxBwYTlNYyNl5pX95bXrB0qr+AZm9bMb31ght721Ug5kHvP17gEAgHpGwIVlnd2tqSyacobcckYHCQkOkk/X7Zdz/7VEZi3bwSA0AAAsjIALS4sKD5WpY7rLvNtPk75J8WZRiPs//kUuff572bA309e7BwAA6gEBF7bQo2WsfDh5qPx9XE+JjgiVNbvS5YLnlsqMT3+VnOOshAYAgJUQcGEbWqZw7ZB2svjuM2V0r+amTOHFb7aZQWhfbTro690DAABWCbj5+fmyePFiWbp0aaVtsrKy5KuvvpJly5ZJQUFBjdsAqllspDx/9QB5+dpTzCC03UdyZeJry+X2t1fJwSwGoQEAEOhCffXGRUVFMm3aNHn99ddNyG3durWsWLGiXLv58+fL1VdfLe3atZPMzEwJCgqSTz/9VLp27VqtNkBZw3s0kyEdG8uTi36TV7/bLvPX7pMlvx2SP4/uJlcMbGOmHQMAAIHHZz24hYWFEh4eLj/++KNMmDChwjZpaWkmuN59992yZs0a2bJli3Tr1k2uueaaarUBKtMwIlT+en4P+eT206RP6zjJOlYg981dL+NfXCab9mdx4gAACEA+C7hhYWHyt7/9TVq2bFlpm48//lhyc3Plj3/8o3kcHBws//d//yfLly+XjRs3etwGqEqvVnEy97ZhMu2CHtIwPERWphyRsc98K49+vlGO5RdyAgEACCA+K1HwxM8//ywdOnSQmJgY17Z+/fq5ntOeWk/alJWXl2duTlrWoLRUQm/1zfke3ngvf+Pvx371qa3l3K6J8o8FG2XRrwfl319vlXk/75W/X9hDTuvU2NLHXl/setx2Pna7Hrfi2O133bnm4rVrXp338euAm56eLgkJCW7b4uPjJSQkxDznaZuyZsyYIdOnTy+3feHChRIVFSXesmjRIrErfz/28+NF2nQNkjnbg2WXDkJ7Y6UMSCySi9sVSUyYtY+9vtj1uO187HY9bsWx2w/XvP7l5ORYI+BqjW7Zg9GeV2f9rqdtypo6dapMmTLFrQc3KSlJRo4cKbGxseKNv0D0F2HEiBGmVMNOAunYx4jI7XkF8tTiLfLfH3bKytRg2ZIdLveO7CK/69+q2oPQAunY65Jdj9vOx27X41Ycu/2uO9d8kdeuufMT94APuO3bt5ePPvpIHA6HmRlB7dq1y/Wcp23KioiIMLey9OJ485fS2+/nTwLl2BuFhcn0cb3l0gFJMvXDdfLL3ky57+MN8vHP++Whi3tJ52YnSmOsdux1za7Hbedjt+txK47dftedax7mlXMcMPPgnsyoUaPk0KFDbnPkzpkzR+Li4mTw4MEetwFqq0/rePn498Pkr2O7S4OwEPlpR5qMeeZb+dfCTQxCAwDAz/i0B/ebb74x3c3bt2+XjIwMM5+tGj16tKmhTU5ONtN9XXnllfLXv/7VTAmmtbNPPPGEREZGmraetAHqQmhIsNx0egcZ1au5TPv4F1m88aA8+78tZhDaQxf3lmGdEjnRAADYPeC+++67kpKSYr7WRRleeOEF8/Xw4cNNwFWvvfaavPzyy/K///3PlBV88MEHMnbsWLfX8aQNUFdaN4qSl687RT5fv1+mffKL7DicI1e9/KNc0r+V3DemuzSOLl/+AgAAbBJwZ86cWWUbDbq33HKLudWmDVCXtN57dO8WMqxzovzri00y64cU+XDVHvnfxoPylzHdZfyA1q6acAAA4F1+XYML+LvYSB2E1ks+nDxUujWPkfScfLl3zlqZ8J8fZOuho652hUUO+XF7mqxMDTL3+hgAANQPv55FAQgUyW0aybw/nCavLt0uT375mwmxo5/6Viaf1VE6NW0oD3+6UfZlHNPPG2TW5hXSIi7SrJo2qlcLX+86AACWQw8uUEfCQoLlljM7yqK7zpSzujaR44VF8vTizfKH2WtKwu0J+zOOyeQ3V8nn6/dx/gEAqGMEXKCOJSVEyWvXD5RnJvSTytaCcBYoTJ+3gXIFAADqGAEXqAc6wKxJTKScrNRWn9Ke3ReWbJG96blmsRIAAFB71OAC9eRglntZQmUe++I3c4uPCpPuzWOlewu9xZj7zs2iJSK0eMo8AADgGQIuUE+axni20EhSowamJ1dnYFi27bC5OYUGB0mnptFuoVdvicy1CwBApQi4QD05tX2CmS1BB5RVVHyg5bnN4yLl63vOloKiItl84Kj8ui9TNuzLNPe/7suSjNx82bg/y9zmrj7xvU1jIlxht0fLWOnRIkbaNW5oVlsDAMDuCLhAPQkJDjJTgelsCRpmS4dc59gzfV7bhQSHSK9WcebmpDW52rO7YW9J4N1fHHp3HM6Wg1l5cjDrkCz57ZCrfURosHRtHmPKHDT0avjt1iLGzNULAICdEHCBeqTz3D5/dX8zW0LpqcKaezAPrg5UaxnfwNyG92jm2p6dV2B6dIt7eYt7fDftz5Kc44WydneGuZXWulED6VHS22t6fFvESlJCA1ZaAwBYFgEXqGcaYkf0aC7LthyUhd/+KCNPHyRDOjU1Pbc10TAiVAa0bWRuTkVFDklJyykOvM4e332ZsjfjmOw+kmtuCzcccLWPiQg1vbulQ6/2/kaGMaANABD4CLiAF2iYHdQ+QQ7/6jD3NQ23lQkODpL2iQ3NbUzvE73C6TnHS2p6T/T4aq1vVl6BLN9xxNxcrxEk5vuddb3O4Kv1vtqbXBOllyhuvD2tVsEeAABPEXABC4uPCpehHRPNzSm/sEi2Hioe0KbB19njezj7uGw9lG1u89eeWGEtoWF4SYnDiR5fndlBV247GV2l7URpBksUAwC8h4AL2IwG027NY83t4uQTA9oOZeWZ3t7SPb7bDh2VtOzjsnRLqrk5hYcEu01f5qzxbdQw3BVudXBd2dkjnEsUa13yyeqPAQCoDQIuAFOC0DQ20tzO6trUdUaO5RfKbwdO9PI6g6+WODjDcGk6LVq35jHy0460CqdG021aoKA9u1qXTLkCAKA+EHABVEoHnfVpHW9uTtrbq4PWTszXWxx0d6XlmnKE0rNFnGyJ4mcWb5ZhnRIlMTpcmsRESHREKDM7AADqBAEXQLV7e5MSosztvJ7NXduzjhUvSvHu8l0yZ+XuKl/n6cWbza30PL4adPWmK7WVvm9SEoKdj6PC+acLAFA5/isBoE7ERIbJwHYJUlDo8Cjgdm0WLXkFRZJ69LgczSswXzunNKtKw/AQSTTBt3wYdvYIO7f5auozZpAAAN8h4ALwyRLFn955hqsGN+d4gaRmHZdDR/PMYLfUiu5Lvj6WXyTZxwsl+3COpBzOqXJ/YiJDi4NwSSA+EX7de4UbN4yQ8NC6WeqYGSQAwLcIuAB8tkSxk5YctGmst6iTvrbW/2q4rTQEZzm/Pm7ujxcWSdaxAnPblppd5b7HR4UVB95KeoX1sc4LrFOnhVYyTRozSACA7xFwAfjVEsVV1f/qYDS96aIUVYXhzGMFFYRf915h7TnWrwuKHJKek29uWw4erWI/RBKiwsuF4MbR4fL819uYQQIAfIyACyAgliiuSRiOaxBmbjpn78noUsfpufnle4OPuvcI6y0tO0+KHGIWxtCbSJbH++ScQeLyF5dJu8SGrv2LjQyVuCi9DzuxreReB9/VdCU5X6D2GIA/IOACCNgliutyqWMtO9Bbl2YxVQY4XfyiotKINbuOyIqU9Crfb0WKtjuxTPLJ6KIaxWE31BV63YNwaKmgfCIY631MRKg5Nm+h9hiAvyDgAkA1aEh3DlTrXqbSYtnWw3LFSz9U+Ro3DGsnjaMjJPNYvmTm6q1AMnLzzU23mfvcfNNTrHXEGqL1Vl3a8asht1zvsH4ddaL3WMOwMxiXDtDVGXRH7TEAf0LABQAvzyBx31j3QXaVlU1kHy8OvmUDcHEozi+1rdTzJfc67ZrDIeY5vYlUPf1aWZFhweVCr7N3ONZZWlHSU/yXj9bbvvaY8gzAfxBwAcCHM0hURksLdG5hvUmj6u+LLrPsDMMZuQXFobikdzgjp9TXpQK0c5vOOlH8GkVyLD9PDmRWv/e4otrjkU8sMdO1NQgPkQZhxbfIUl/r9kjX18HFz5d6zvW41Nf+UvZCeQbgXwi4ABAAM0hUl4Y/vTWNiaxRT+TRkl5h9yBcutf4RHDekZotKWlVz0m8NTXb3OqSllGUD8jB7oG4TCgu3eZkAdq5vaqBfpRnMLgQ/oeACwAWm0GitnQ/TY1uVJhH7T2tPb7nvC7StnFDyT1eaHqYc/V2vMjcm8fHS7ZV9NjtuSLXax4vKDI3Dd31RbNtZGjpEHwiQGv4Xb7jSKXlGepPH6wz+xyl31MqVOvrnAjc/tUjXR127r2mLMV/EXABwMYzSHiz9vjWMzvVyXnQ+mStMXYGXvfAXD4ga7isKjSXfQ39Hh3gp7SW2dmuJjR8//HdNR7PmhGhAdot+Oq2UuG6VE906TYmPIcWl31Eas92mTAdURLSndvq4lrYuffazsE+EMI9ARcA4De1x57Q+mTTgxoeIvWpoLBIjmmQriBAOwPyd1tT5c0fdlb5Wp2bRkvDiFDzOsfKhG4N604aqp0r8NW3sJCg8iFZe6VPEqZLb9Pe60c+23jS3uu/ffyLdG0WK+FhwRIaHGR+BvReVwIs/TiQ5nq2e7APlHBPwAUAWKb2uC5pCIvWW0Tl/6mMjwr3KOD+fVwvGdKxcZU90s4g7RaENWAXlPQsFxRJnrOX2WzTEF4ctl1tSoXnk4Xp/EKH5BcWL2VdX3R+6LP/9XWV7fRvn9DgYFfgDQkpCcLObSEnwnBIcNmwXKqdc3uIh+2cr1fq9Yvf3/17i++Lv1cnz5v64boqg3235rGushP9Y8IZ6sNCgs3xBlqoD7RwT8AFANSJQK89rs/yDG3n6x7pmoRpE5zzKw7TOrhww76qV/ILDzFpzvSI69zOFe5XyZzPUrMqEL+jwf6sx08e7E3oLQnXGnpNEHb2buu2kkBdNhy7tpX93jLbQkv3lpe8XnHId38917ay31vB6+n/7v/4l4CYEpCACwCoM3aqPfZFeUZt1WWY9nRw4Rs3DHL1XmvALnQ4TP1mgX5dqPdFJx4Xaa+y++Pi+yIpKDzxuKDMY7d2zsclr33ifSppZ9q6Pz7Rvnw7XXRlZ1rV80prIFT6fRUp7kEvFKm/8ZFe55wS8Cetya3kEwtvIeACAFALVizPqK/eaw3YwaI9ghKwPA32/72xONg7HM5w7JB8DfMl9yaAO7eVBHtXKC8J2G7bnK9REsadobzA7fVKthVV9Xontrtez/mapV7PtV8l99qbr8G8KgezTvwe+AoBFwCAWrJjeUag9V77Kthrra2WD2iobyABnOzF83Bfk/m365rnC40DAIAqyzMGJNqjPKN077UGutL0sb8MNqqvYK/KXmErB/vS4b6yI9PtLaqoOfcWenABAECN2bH32q5lKSEB1GtPwAUAALVit8GFdg32gRTuCbgAAAA1YMdgHyjhnoALAAAAS4V7BpkBAADAUgi4AAAAsBRLBNzMzEyZMmWK9O3bVwYNGiRPPPGEFBWdWG8bAAAA9mGJGtyLLrpI0tPT5emnn5YjR47ITTfdJKmpqfLwww/7etcAAADgZQEfcL/++mv56quvZP369dKzZ0+z7eDBg3LnnXfKn//8Z4mNjfX1LgIAAMCLgq0QcFu1auUKt2rMmDGSl5cny5Yt8+m+AQAAwPsCvgd3165d0rx5c7dtzse7d++u8Hs0/OqtdA2vys/PN7f65nwPb7yXv+HY7XfdueZcczvh552fdzvJ93Keqc77BDkcjtIrrQWciRMnysaNG916a3WAWVhYmDz//PMyadKkct/zwAMPyPTp08ttf/vttyUqKqre9xkAAADVk5OTI1deeaVkZGRUWYIa8D24iYmJcvjwYbdtOtBMQ64+V5GpU6eaWRdK9+AmJSXJyJEjvVKzq3+BLFq0SEaMGGGCuJ1w7Pa77lxzrrmd8PPOz7ud5Hs5zzg/cfdEwAfcgQMHmmnBdGBZ06ZNzbalS5ea+1NOOaXC74mIiDA3J2cndm5urlcukP5A6F8h+n4FBQViJxy7/a4715xrbif8vPPzbif5Xs4z+j7Kk+KDgC9R0BPbpUsXGTVqlLzwwguSnZ0tw4cPN2F3wYIFHr2G1upqDy4AAAD8f/xV69atrR1w1erVq+Wqq64yQVUHj51xxhmmnrZJkyYefb+WM+zdu1diYmIkKKj+11J2lkToBbLbNGYcu/2uO9eca24n/Lzz824nmV7OMxpZs7KypGXLlhIcHGztEgWVnJwsGzZskH379pnSg4SEhGp9v56kqv4SqA/6w2CnoFMax26/684155rbCT/v/LzbSawX80xcXJxH7SwRcJ1atGjh610AAACAjwX8Qg8AAABAaQRcH9AyimnTprnN5GAXHLv9rjvXnGtuJ/y88/NuJxF+nGcsMcgMAAAAcKIHFwAAAJZCwAUAAIClEHABAABgKZaaJixQ6IIUO3bskH79+kl0dLTYRUZGhmzbtk1atWrlWlbZLg4cOGAWE2nTpo00btxY7Gb//v2yZcsW6dSpkzRv3lysTBebWb58ebntPXv2lEaNGokd6NCOjRs3mjnGu3btKnaY7H7t2rUVPte9e3db/M6npqZKSkqKxMfHS/v27auchN8qCgsLzX/P9b9v+jvuj4Ot6vIa6+91t27dJDExsdJ/63XRhw4dOvj+514HmcE7vv/+e8eFF17oSExM1IF9juXLl9vi1G/atMkxbtw4R6NGjRz9+vVzREdHO84//3xHWlqaw+pWrlzpOP300x2tWrUyxx4ZGemYMGGCIzc312EXOTk5jl69ejmCgoIczz//vMPqtm/fbn6/BwwY4Bg2bJjr9t133znsYPHixY527do5kpKSzM/84MGDHSkpKQ4rW7t2rdu11lunTp3Mz8GSJUscVnbs2DHH5Zdf7oiKinL079/f0bx5c0fnzp0dK1ascFjd0qVLHR07djQ/6z179nQkJCQ4PvzwQ4fVbNiwwXHNNdc4WrRoYX6mZ8+eXa5NQUGB48YbbzT/jevRo4cjIiLCcf/99zt8iYDrRS+++KJj7ty5jvXr19sq4H766aeOjz76yFFUVGQeHzp0yNG1a1fzC2N1H3zwgQm5Tjt27DBBf8aMGQ67uPnmmx133nmn+QfPTgF38+bNDrvRf9v0Oj/88MOubfrzv2zZMofdXHvttY4OHTq4/t2zqmeeecbRsGFD83Ov8vPzHRdccIH5A8/qf7g3bdrUMXnyZEdhYaHZpv9914DnPBdW+u/YG2+84cjKyqo04D711FOO+Ph406Glvv32W0doaKjj448/dviKPT5D8BOTJk2Siy66SEJCQsRORo8eLePGjZOgoCDzWD/aGD9+vCxdulSs7pJLLpH+/fu7Hrdt29Z8fLdnzx6xg/fff1+WLVsmjzzyiNiNfmy5evVqs266XTz00EPm48upU6e6tunP/+DBg8VOtGRhzpw5ctNNN7n+3bOqQ4cOmVVE27VrZx6HhobKqaeearZb2fr16+XgwYMyefJkVzmG/vc9ISFBZs2aJVb779i1114rkZGRlbZ59dVXZcKECdKlSxfz+LTTTpOzzjrLbPcVAi58QmsUtR7TDgoKCkyYX7Rokdxzzz2mHvf2228XOwS8P/zhD/LWW2+d9B9Gq9L/IFx99dWmDk2DTk5Ojljd4sWL5fzzz5fc3FxZuXKlqcWzo9mzZ8vx48fl+uuvF6u7+eabTc313XffLV9++aUJNC+88IL5Y8fKNMiq0p0V2dnZkp6ebn727SQ/P19++eUXGTBggNt2/UNH/8j3FQaZwetef/11E/b+97//2eLs6z96f/7zn01Png60+uMf/ygdO3YUq4f6K664whx3nz59xE4aNGggc+fONb05Sv/hP/vss6Vhw4by9NNPi5UH22iPloZaHVimwV4HHWmP7jvvvGMGWNrFK6+8YoK+9mxaXevWrU0v5oMPPmj+TdfBtIMGDZLhw4eLlem/4XqNnceug+ueeeYZCQ8Pl7S0NLGTjIwM8/tfdlCZPvbluaAHF141b948ueWWW+S5556TM8880xZnPy4uzvTg/vzzz+ZjrTfeeMPtI1wreuKJJ8yIW/2LXo9db9rLs3XrVsv3bjRr1swVbpWOrL7jjjtMyLMyLb3Sj2oXLFgg33zzjem52blzpxQVFZnfebtYt26d+YRKezbt4NFHHzUlSKtWrTLXXGcJ0j/mRo4caa69lWkZiv5uaynWzJkzzSc2+ses/pFrJ2FhYeb+2LFjbtv1kxwN/L5CDy68Rv/Dp7W3jz/+uPmr1460/lbPwWeffSaPPfaYWPkfPA16pYO8foylPZvaw2f1sFeWngvt3dSeba1RtCqtMR8yZIirHlOnQbzqqqss/wdd2d7bpKQkGTVqlNjB/PnzZcyYMebfNufvvo43Oeecc0yZkk4XZVU6Jdhdd91lbk5//etf5dJLLxU7iYuLMz3YZceW6GNffnJDDy68QgOd/tLrX/tal2kXWp5QlpYp+Hx+wHqm/+A7e26dN/1L/v/+7/8sH24ruuYLFy40H9tbOdyq8847r9x/5LRHr0mTJmIHOgfym2++KTfccINt5oHVa6vXuDRn7XVlc6VahfZQlu3E0WO/5pprxG6GDx9uPqF10j/m9XyMGDHCZ/tk7X9t/cy+ffvMR7T6sZ3Sj6y1S1//8tXFD6xKP668+OKLXTMKOGdP0P8ADB06VKxMj1uPceDAgaZG6eOPP5YvvvjC/OLDmvSPOP0d1x48/aj2gw8+MNddP8a0Ou2p1bIU/QNHZ0/Rj+ufffZZU7JiBx999JEcOXLEBFy70AGzGmLuvPNOGTt2rOm1/dvf/ibXXXedxMbGipVNmzbN9OKefvrpsmHDBnnggQfMtuTkZLGS9PR0U16n/w1TmzZtMv8d1xpz53iS+++/38yWop/Oao++luJpvpkyZYrP9jtI5wrz2bvbzIcffljhP/T6D4ROr2FVOqq2oqlC9B8GHXVtZdqb9/zzz8u3335ralC1F+/WW2+1/CCzipx77rmmXk2njLMyvc7vvfeeCbX6HwadNkf/0bfDil5KVyvUMiT9j2DLli1NXaL27NrBfffdZwbc6BgDO1mxYoX85z//MeFWV+vT3ryJEyda/hMLnSnjqaeeMoPrdFYF/VnXcGc1P/74o5kloywda6Cfyjlpp92TTz5p/sDXf+/uvfdeV+mKLxBwAQAAYCn2KBICAACAbRBwAQAAYCkEXAAAAFgKARcAAACWQsAFAACApRBwAQAAYCkEXAAAAFgKARcA6tjq1avN4h7+Jj8/X77++mt59913Zfv27TV+HV2hTCe3BwB/Ze1lRgDYjq6gpQGzd+/e0rNnT7dwp8vmnn322dKsWbN63QddpnLLli1mCU9/oWvDDxkyxJyH7t27S/PmzU+6ypAuPbp582aJiYkx7XVZTicNyLpU5znnnFMn+7Z27VpJS0uTs846q05eDwAIuAAsZd68eXLPPfeYpSJ1/XTncqG6bPIVV1whixYtqveA649Wrlwpq1atkqysLGnYsOFJl1294YYbZP/+/XLqqaea5Ug3btwoAwcOlJdeesksSVrX3n77bfO+BFwAdYWAC8By4uPj5eDBg/LKK6/ILbfcUmGbH374QYKCgmTQoEFuvZZ79+6V4cOHm8faE3z06FE55ZRTZM2aNZKamiqnnXaaNGrUSHJycuS7774z7YYOHVphaNQ2+hpHjhwxvblxcXHl2vz666+m17lly5aSnJwsYWFhruec7z9gwAD5/vvvzde6/ntFHA6HLF++XHbv3m16ZvW1nDTof/jhhybs6x8A6rLLLpPgYPcqNQ2y2sN97bXXmjXlw8PDXc99/PHH5o+EigKuJ+dS7dmzx6xXr+dBjykyMtIcu56DAwcOyDvvvGPa6Tlu3bp1rc5PRe8FwD4IuAAsR8PmHXfcIQ888IBcffXVFYbP5557zgS+0qFMQ+D8+fNdoUxLDT7//HPTi9mlSxcTHvft22fC3/Tp0802rWUtLCw04VKDtdO2bdukX79+kpSUJIcOHTJhT19Lw7I6duyY2TcNybpNX0dDogbQdu3aud5/4cKFJrxq4NPgWlHATU9PlzFjxsiOHTvMe/7000+m93Xu3LkSERFhguuyZcvMfn700Ufmey699NJyAVfPV2JiYrlwq8aNG1fp+fbkXD7zzDNy3333ybBhwyQvL8/8AaKlDrrPWgqhJQrOfdPj1P2o6fmp7L169ep10p8bANZBwAVgSRpwn332WRPW/vrXv9b4dbZu3WoCo/YeakDs27evTJo0yXzkrzW+WtOq5RCvvvqqTJkyxfV92vOoAVMDlwYw7RW99dZbzUfxatq0aabXUoNwgwYNTJsbb7xRbr/9dhMMnTScLlmy5KT1vBq2tZdYe001ZGuY7t+/vzz99NNy7733yu9+9zvTg6n3zl7SinzxxRcmVJYNt3VBw7Oeo/Hjx5vHKSkppkf8/PPPN/W8el5K79uf/vSnGp+fyt4LgH0wiwIAS9JA9/e//10effRR04NaU9oT6vy4PyQkxNSias+gcwCbfmSuH4H/9ttvbt/XqVMnV2+r9jxq0NRQrIFZvfbaa2Yg3IIFC+T99983Nx34pbMcaJhz0h7ZqgaraTC87bbbXD3I+nH+xIkTTxpmy9Jeau0JbtOmjdTX9fjll1/MHwmqbdu25rxVpjbnp7rvBcB6CLgALOu6664zge0f//hHjV9D621L04/8K9qmJQelOT9Gd3LOWKC9iVqbq6Fbe1znzJljZnfQj/S1t1J7NDVsOpWevaAi+hG8Dgjr0KGD2/aOHTua9/KU9trq7fDhw1IfZs2aZQJ3kyZNTPB/66233IJqabU9P9V5LwDWRIkCAMvSGtMZM2aYelMNu2WfKyoqcttWNqTWhpYMVPRYa0u1h1HD5JVXXmnKHU5Ge39PRsN1bGysqWEtTR/re1WH9k7rgLHq8uRcai2ulhNoLa3WIt91111m8Jj2spdV2/NTnfcCYE304AKwtAsuuEAGDx5sBh2V1qpVKzNXbWn68Xdd0VkXdACVk/ZAao+iDkzTQDhixAgz7ZbzY3QnHf1fXVoyoa/vpL2V2uupsxFUh9YQaz2r1g6XlZGRIZmZmRV+X1XnUuuUdaCXsyd78uTJZso2Z5iOjo52C8S1OT9VvRcAe6AHF4Dl/fOf/zRTeZWmvYOPP/64GbSkdZw6cEnraMt+1F9T2qs6atQo8/o6WOqxxx6Tp556yjVdlX595plnmhCq+6I9oBouNezpR+zV8cgjj5jju+qqq8xcsp988on5OF8/3q+OSy65xJRzaCDUm+6blgNoPeunn34qixcvNsdVVlXnUkOn7t/o0aPN81p+oDMg6DlROkvCww8/bGY/aNq0qXnfmp6fqt4LgD0QcAFYSrdu3cpNaaUreN1///3mY2odqKR0AJPOnfrmm2+a+VK1hEFvuhiCk85EoAO2yn6Mr4sllH390lNuOb9Pt+v8sVqeoIOktDe59CA0nZ/29ddfN++pA8Suv/56ufDCC0/6/hXp06eP6THWeX91iWD9vhdeeMH0rDrp1zr3bVV0xgmdfUD3V2c30ECp02tpAHWGW32/0jMtVHUuo6KizP7psep0Zfo62kusc+4q/UPg5ZdfNgFWX0d7XnXKsZqcn6reC4A9BDmovAcAAICFUIMLAAAASyHgAgAAwFIIuAAAALAUAi4AAAAshYALAAAASyHgAgAAwFIIuAAAALAUAi4AAAAshYALAAAASyHgAgAAwFIIuAAAALAUAi4AAADESv4fBH53SGKqXEUAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 5))\n",
    "\n",
    "plt.plot(\n",
    "    range(1, 11),\n",
    "    wcss,\n",
    "    marker='o'\n",
    ")\n",
    "\n",
    "plt.title(\"Elbow Method\")\n",
    "plt.xlabel(\"Number of Clusters\")\n",
    "plt.ylabel(\"WCSS\")\n",
    "plt.xticks(range(1, 11))\n",
    "plt.grid(True)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "9dc0d7f4-f6e4-469b-bbaa-647ac4854092",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sneha\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1419: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "kmeans = KMeans(\n",
    "    n_clusters=3,\n",
    "    random_state=42,\n",
    "    n_init=10\n",
    ")\n",
    "\n",
    "clusters = kmeans.fit_predict(X)\n",
    "\n",
    "data[\"Cluster\"] = clusters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "f5b90fc4-7967-4a43-b671-2b0f5091c73d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cluster assignments:\n",
      "   Id      Species  Cluster\n",
      "0   1  Iris-setosa        1\n",
      "1   2  Iris-setosa        1\n",
      "2   3  Iris-setosa        1\n",
      "3   4  Iris-setosa        1\n",
      "4   5  Iris-setosa        1\n",
      "5   6  Iris-setosa        1\n",
      "6   7  Iris-setosa        1\n",
      "7   8  Iris-setosa        1\n",
      "8   9  Iris-setosa        1\n",
      "9  10  Iris-setosa        1\n",
      "\n",
      "Number of data points in each cluster:\n",
      "Cluster\n",
      "0    62\n",
      "1    50\n",
      "2    38\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(\"Cluster assignments:\")\n",
    "print(data[[\"Id\", \"Species\", \"Cluster\"]].head(10))\n",
    "\n",
    "print(\"\\nNumber of data points in each cluster:\")\n",
    "print(data[\"Cluster\"].value_counts().sort_index())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "f3480030-a966-4c1d-a96c-1792587ca613",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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xi2DWUshbOmcoKCgPD+erO29atUqdO/e3W7hcGLfFyV877Pv3RXf++x7d5XoxJhHFsGRwUd7C5i4ZTBrSS2QQDa/glk/Pz91LAaz+Yt9X7DY/+x7d8X3PvveXSXmQcxjLyWUE8CIiIiIyGUxmCUiIiIil8VgloiIiIhcllvmzDpavishIcFp+SMeHh6Ii4uDyWRyyj6JiIiIiMFsliSIPXPmjAponVUnrWzZsqp6Auva5i/pe5kFKT+JiIio6OHIbAYS9Fy+fBlGo1GVg7BVpNdREhRHRUUhICDAKfsjx19L6ff4+Hhcu3bNZo06IiIick0MZjNISkpCTEyMWm1CSks4M2XBx8eHwWw+8/b2VukdUqtOUjzkSwoREREVHRwmzMCS0+rl5VUQrwflActrKbnLREREVLQwmLWCua1FB19LIiKiootpBkREROR0uukmELcIetI5QPOC5tUa8O4ITcu7dC898RDMUSsA1IQ56mvogb2gedRMv40eC8Quh550UF3XPBoAvr2gab551i4qwsFsZGQkpk+fjq1bt6rSVe3atcPjjz9u8xT/lClT8Mcff6S7rVq1avjuu+/yocWUl6Onq1evxp133ml1G5lA9+uvv+Kee+7hC0FEVEjpugl61KdA9AyZNZJ6EliP+QkwlAWKfQbN6zbnHtN0Ffqt0UDiHiBJ5rtMUMfX47+C7tUBWrFPoBmCocf+Dj3iHUCPSg2BdCQBkROBwNeh+fVzaruoiAezMimqUaNGuP/++9GvXz816erdd9/FggULsHLlSqsTdY4cOaImU7355pvpghwCbty4gffeew9LlizBhQsXULJkSTRr1gwvvvgiOnXq5NQukoCyZs2amDx5MrueiIhS6RETgdhZaT/x0/x6DXroo0CJOdA8Gzul13TzLeihgwDTpZRbTOl/JmxRx9T9BgERr6d5ZFKanURDjxinfmVA63o8CnIkbu/evShevHjqbRLc3n777di9ezdatmxp9bGlS5e2OYJXKMhEsk2bgMuXgTJlgCZN8vRwFy9eRJs2bVCjRg3MmjULTZs2RVhYGP755x+88847aN++faGeyc86sERErk9PPJYhkM1IAtskFfBqJeY655gyAmy6mD5oTscEJB0GZETW3r4i3wV8ezLlwMUYCjKYTRvIihIlSqif0dHRNh/7999/o3fv3hgyZAi+/fbb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MJlMefoYIiIiIipaCjSYlUB01KhRCAkJQePGjVG2bFn8+uuvTn8MERERERVNBRrMfvjhh5g3bx727duH0NBQTJw4EQ8//DAOHjzo1McQERERUdFUoMHs1KlT8cQTT6BeveRlUYcPH47q1avj22+/depjiIiIiKhoKrBg9urVqzh//jxat05f+qlt27bYvXu30x5DREREREVXgdWZvXHjhvpZImUFLIuSJUtiy5YtTnuMiI+PVxeLiIgI9TMxMVFd8prlGPlxLGLfFyZ877Pv3RXf++x7d5XoxJjH0X0UWDBrSFlGNWNDExISYDQanfYY8f7772PChAmZbl+1ahX8/PyQX1avXp1vxyL2fWHC9z773l3xvc++d1ernRDzxMTEFO5gtmLFiurnlStX0t0u1y33OeMxYty4cRgzZkzq9fDwcFSuXFmlKwQGBiKvSfC9bt06dO7cGZ6eyUvGUv5g3xcs9j/73l3xvc++d1eJTox5IiMj1U9d1wtnMCtB5G233YaVK1fioYceSh1hXbNmTbrA8/r164iLi0OlSpUcfkxG3t7e6pIxzaBatWp5+AyJiIiIKLckqA0ODi58wawYP348+vbti6ZNm6pR0k8//RReXl546qmn0o2qbt++PbX0liOPsad8+fJqIpkEx5qmIa9J8CzBuBwzKCgoz49H7PvCgu999r274nuffe+uIpwY88iIrASyErfZUqDB7L333ov58+fj888/V6W1GjVqhI0bN6ab4FW6dGmVEpCdx9gjube20hLyiryoDGYLBvu+YLH/2ffuiu999r27CnJSzGNrRNZC0+0lIpBTvqXIiyG5ugxm8xf7vmCx/9n37orvffa9u4oogJinQBdNICIiIiLKDQaz+UAmn0mub9pJaJQ/2PcFi/3PvndXfO+z792VdwHEPEwzICIiIiKXxZFZIiIiInJZDGaJiIiIyGUxmCUiIiIil1WgdWbdQVJSEnbv3o3ixYujTp06Bd0ct+v7Y8eOwcfHB1WrVoXRaCzoJrmVqKgoHD9+HKVKlVIFtCn/SWmcAwcOoEKFClzxMB+YTCZs27Yt0+3yt1/+HVD+MJvNOHr0qFpKtVatWuz2fCCrtcrnbVZkkauAgIA8PT4ngOWR6OhoTJo0CT/++CNu3bqFnj174tdff82rw1GGdaHffvttTJs2DWXKlFEf6PJH7fvvv0eX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",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 5))\n",
    "\n",
    "plt.scatter(\n",
    "    X[\"PetalLengthCm\"],\n",
    "    X[\"PetalWidthCm\"],\n",
    "    c=clusters,\n",
    "    s=50\n",
    ")\n",
    "\n",
    "centers = kmeans.cluster_centers_\n",
    "\n",
    "plt.scatter(\n",
    "    centers[:, 2],\n",
    "    centers[:, 3],\n",
    "    c='red',\n",
    "    label=\"Centroids\"\n",
    ")\n",
    "\n",
    "plt.title(\"K-Means Clustering of Iris Dataset\")\n",
    "plt.xlabel(\"Petal Length (cm)\")\n",
    "plt.ylabel(\"Petal Width (cm)\")\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "265e1f56-662c-482f-b70c-989366961511",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56b3581d-cb5f-4ae4-8661-9b4714bf73f6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "10f8f579-5441-4b5a-9f18-a38d4842af7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Random Forest Classifier for Car Evaluation\n",
    "\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score,\n",
    "    classification_report,\n",
    "    confusion_matrix\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "bcf798f0-4315-4890-a179-645dcb62c133",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "First 5 rows:\n",
      "   vhigh vhigh.1  2 2.1  small   low  unacc\n",
      "0  vhigh   vhigh  2   2  small   med  unacc\n",
      "1  vhigh   vhigh  2   2  small  high  unacc\n",
      "2  vhigh   vhigh  2   2    med   low  unacc\n",
      "3  vhigh   vhigh  2   2    med   med  unacc\n",
      "4  vhigh   vhigh  2   2    med  high  unacc\n",
      "\n",
      "Dataset Shape:\n",
      "(1727, 7)\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv(\"car_evaluation.csv\")\n",
    "\n",
    "print(\"First 5 rows:\")\n",
    "print(df.head())\n",
    "\n",
    "print(\"\\nDataset Shape:\")\n",
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "de257e7f-6858-4e4b-9a02-8de1fd268ad9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Column Names:\n",
      "['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']\n"
     ]
    }
   ],
   "source": [
    "df.columns = [\n",
    "    \"buying\",\n",
    "    \"maint\",\n",
    "    \"doors\",\n",
    "    \"persons\",\n",
    "    \"lug_boot\",\n",
    "    \"safety\",\n",
    "    \"class\"\n",
    "]\n",
    "\n",
    "print(\"\\nColumn Names:\")\n",
    "print(df.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "6eff838f-f270-48d9-b488-b0ffa601fb62",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Missing Values:\n",
      "buying      0\n",
      "maint       0\n",
      "doors       0\n",
      "persons     0\n",
      "lug_boot    0\n",
      "safety      0\n",
      "class       0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(\"\\nMissing Values:\")\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "d9f408cd-efbe-4ea0-a178-21f4d708ecc8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features:\n",
      "  buying  maint doors persons lug_boot safety\n",
      "0  vhigh  vhigh     2       2    small    med\n",
      "1  vhigh  vhigh     2       2    small   high\n",
      "2  vhigh  vhigh     2       2      med    low\n",
      "3  vhigh  vhigh     2       2      med    med\n",
      "4  vhigh  vhigh     2       2      med   high\n",
      "\n",
      "Target:\n",
      "0    unacc\n",
      "1    unacc\n",
      "2    unacc\n",
      "3    unacc\n",
      "4    unacc\n",
      "Name: class, dtype: object\n"
     ]
    }
   ],
   "source": [
    "X = df.drop(\"class\", axis=1)\n",
    "y = df[\"class\"]\n",
    "\n",
    "print(\"Features:\")\n",
    "print(X.head())\n",
    "\n",
    "print(\"\\nTarget:\")\n",
    "print(y.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "ce0e8d09-e0ea-4d2c-99cb-e202afbdd66a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Encoded Features:\n",
      "   buying  maint  doors  persons  lug_boot  safety\n",
      "0       3      3      0        0         2       2\n",
      "1       3      3      0        0         2       0\n",
      "2       3      3      0        0         1       1\n",
      "3       3      3      0        0         1       2\n",
      "4       3      3      0        0         1       0\n"
     ]
    }
   ],
   "source": [
    "encoder = LabelEncoder()\n",
    "\n",
    "for column in X.columns:\n",
    "    X[column] = encoder.fit_transform(X[column])\n",
    "\n",
    "# Encode target variable\n",
    "y_encoder = LabelEncoder()\n",
    "y = y_encoder.fit_transform(y)\n",
    "\n",
    "print(\"\\nEncoded Features:\")\n",
    "print(X.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "2a82187c-e55a-4b2f-bd7c-1b0f5aaf4c7d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Data Shape:\n",
      "(1157, 6)\n",
      "\n",
      "Testing Data Shape:\n",
      "(570, 6)\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X,\n",
    "    y,\n",
    "    test_size=0.33,\n",
    "    random_state=42,\n",
    "    stratify=y\n",
    ")\n",
    "\n",
    "print(\"Training Data Shape:\")\n",
    "print(X_train.shape)\n",
    "\n",
    "print(\"\\nTesting Data Shape:\")\n",
    "print(X_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "3be457c8-b27d-41bb-8004-a8e38ebda1da",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = RandomForestClassifier(\n",
    "    n_estimators=100,\n",
    "    random_state=42\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "509cd650-cf48-42d2-b050-ef2e94f78f60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest model trained successfully!\n"
     ]
    }
   ],
   "source": [
    "model.fit(X_train, y_train)\n",
    "\n",
    "print(\"Random Forest model trained successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "52db52b2-7a63-43f9-8cf5-8301a1a2e79c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions:\n",
      "[2 0 3 2 2]\n"
     ]
    }
   ],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "\n",
    "print(\"Predictions:\")\n",
    "print(y_pred[:5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "a48737df-fb95-455e-83c0-3c3de3fcf1de",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.980701754385965\n",
      "Accuracy Percentage: 98.0701754385965 %\n"
     ]
    }
   ],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Accuracy Percentage:\", accuracy * 100, \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "b63f7e29-c484-4dec-a13d-de537f7b1db7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "         acc       0.96      0.95      0.96       127\n",
      "        good       0.95      0.87      0.91        23\n",
      "       unacc       0.99      1.00      0.99       399\n",
      "       vgood       1.00      0.95      0.98        21\n",
      "\n",
      "    accuracy                           0.98       570\n",
      "   macro avg       0.98      0.94      0.96       570\n",
      "weighted avg       0.98      0.98      0.98       570\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Classification Report:\")\n",
    "\n",
    "print(\n",
    "    classification_report(\n",
    "        y_test,\n",
    "        y_pred,\n",
    "        target_names=y_encoder.classes_\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "9a83ec3d-f94a-4bcf-8153-505880c6a8dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[121   1   5   0]\n",
      " [  3  20   0   0]\n",
      " [  1   0 398   0]\n",
      " [  1   0   0  20]]\n"
     ]
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "print(\"Confusion Matrix:\")\n",
    "print(cm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "43eb1775-28a0-45be-a16e-63dbf9d6d288",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 700x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(7, 5))\n",
    "\n",
    "sns.heatmap(\n",
    "    cm,\n",
    "    annot=True,\n",
    "    fmt=\"d\",\n",
    "    cmap=\"Blues\",\n",
    "    xticklabels=y_encoder.classes_,\n",
    "    yticklabels=y_encoder.classes_\n",
    ")\n",
    "\n",
    "plt.xlabel(\"Predicted Class\")\n",
    "plt.ylabel(\"Actual Class\")\n",
    "plt.title(\"Random Forest - Confusion Matrix\")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "9e1ea034-be3c-4d7b-99cc-9e5555b9e9d0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature Importance:\n",
      "safety      0.273577\n",
      "persons     0.222308\n",
      "buying      0.171535\n",
      "maint       0.167442\n",
      "lug_boot    0.093228\n",
      "doors       0.071910\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "feature_importance = pd.Series(\n",
    "    model.feature_importances_,\n",
    "    index=X.columns\n",
    ").sort_values(ascending=False)\n",
    "\n",
    "print(\"Feature Importance:\")\n",
    "print(feature_importance)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "8a95ff35-1b76-40da-8082-57938de9fbbe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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jAsqihYtesH7Bi2wqFu3rr366qu2c+fO2NSlJUuWuBP52rVrp/gxKdlu5syZ3eIv0oq89aUI9y9GpOLYhieOa3jiuIYvjm14ipTjmvsCIxVBL96+7bbbXKTXp08fV9/wzz//2ODBg12rWF8KkqJBFWJPmzYtyY9Jyn0AAAAApK6gBRZZsmSx6OhoW716tRsd0PCKgojx48cHpCSp4Pro0aNJfkxS7gMAAAAgdQUtFUqqVKliP/30k5vELmPGjJYjR454wy7KXcuXL1+SH5PU++B/lAI2cODAeKlgCH0c2/DEcQ1PHNfwxbENTxzXhEXFJKV3FAAAAACkxVQoAAAAAOGDwAIAAACAZwQWEWjz5s3B3gVcIsp0JNsRSJs0f9Lx48eTfRvStkmTJtm3336b7NsQmtRoCP+HwCIC3XLLLVanTh0bO3asHThwINi7g1Ry6tQp69Gjhx07dsz9PHv2bMubN68VKFDA/R+h+0fryJEjCS7qmMcftdClboVqi56Qd999106fPn3J9wneLVq0yNavX5/gbQsXLrQNGzbwNoeoDz/80D799FP3f/0Ovu6661wRd7t27dzfYBBYRKSPP/7YbrjhBhs0aJBddtllds8999iCBQu4sh3iRo0aZdmyZXOLRikee+wx69mzpz377LPWtWtXfumFqGXLllnOnDkTXNTxLkOGDK4T3nvvvRfsXUUqUKD4888/u6AiEibdiiT79u1zrfALFSoU7F1BCqjTaN++fa1hw4bu5zfffNNdyNM51dq1a23KlCm8rwQWkalixYo2fPhw27Ztm33wwQdumvZmzZpZmTJl3GSCf/31V7B3ESnw9ddfu4BRfv31V3eColbCGsVQy+Y///yT9zUEVa9e3e677z676qqr3FVsHWddMevVq5cLLvTH7Pbbb7cHH3zQFi9eHOzdRRJojqWoqCg7ePCgG1XU/31L+vTp3THXxQCElk6dOrljqHSnzp07BxxXLfnz53fpbb7f0wgtP/74o1155ZXu4p3MmzfPBRotWrRwF/JIcUsD81gguHSls3Xr1m6ZMWOGdenSxZ2IaiSjVatW9sorr1jZsmU5TCFCJyR79+51/58/f741atTI/TETjWD4/o/QouM2d+5clz6hk1D/lEYFjCtWrLBXX33V3e+///0vJy0hQKmJGpHQ797333/fsmfPHvB7uUSJEu5C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",
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 5))\n",
    "\n",
    "feature_importance.plot(kind=\"bar\")\n",
    "\n",
    "plt.title(\"Feature Importance - Random Forest\")\n",
    "plt.xlabel(\"Features\")\n",
    "plt.ylabel(\"Importance\")\n",
    "\n",
    "plt.tight_layout()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "19d4ae43-d76b-4b7c-a4a5-f473dfe15846",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dc9c4981-2e67-4f29-8891-fce7c429a72b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "7b88c329-05ab-4cd2-8eb2-907b3854725e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "d1950291-e82c-4b61-b51b-0abc7af26148",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Maze:\n",
      "[[0 0 0 1 0]\n",
      " [1 1 0 1 0]\n",
      " [0 0 0 0 0]\n",
      " [0 1 1 1 1]\n",
      " [0 0 0 0 0]]\n",
      "\n",
      "Start: (0, 0)\n",
      "Goal: (4, 4)\n"
     ]
    }
   ],
   "source": [
    "# 0 = free path\n",
    "# 1 = wall\n",
    "\n",
    "maze = np.array([\n",
    "    [0, 0, 0, 1, 0],\n",
    "    [1, 1, 0, 1, 0],\n",
    "    [0, 0, 0, 0, 0],\n",
    "    [0, 1, 1, 1, 1],\n",
    "    [0, 0, 0, 0, 0]\n",
    "])\n",
    "\n",
    "rows, cols = maze.shape\n",
    "\n",
    "start = (0, 0)\n",
    "goal = (4, 4)\n",
    "\n",
    "print(\"Maze:\")\n",
    "print(maze)\n",
    "print(\"\\nStart:\", start)\n",
    "print(\"Goal:\", goal)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "7ff53b82-6cfa-4ae7-9daf-6bbba33e202f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Actions:\n",
    "# 0 = Up\n",
    "# 1 = Down\n",
    "# 2 = Left\n",
    "# 3 = Right\n",
    "\n",
    "actions = [\n",
    "    (-1, 0),\n",
    "    (1, 0),\n",
    "    (0, -1),\n",
    "    (0, 1)\n",
    "]\n",
    "\n",
    "# Q-table\n",
    "q_table = np.zeros((rows, cols, 4))\n",
    "\n",
    "# Learning parameters\n",
    "learning_rate = 0.8\n",
    "discount_factor = 0.95\n",
    "\n",
    "epsilon = 1.0\n",
    "epsilon_decay = 0.995\n",
    "epsilon_min = 0.01\n",
    "\n",
    "episodes = 1000\n",
    "max_steps = 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "24b23416-4094-4d8c-b3d8-86330c7770bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_next_state(state, action):\n",
    "\n",
    "    r, c = state\n",
    "    dr, dc = actions[action]\n",
    "\n",
    "    new_r = r + dr\n",
    "    new_c = c + dc\n",
    "\n",
    "    # Check boundaries\n",
    "    if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:\n",
    "        return state\n",
    "\n",
    "    # Check wall\n",
    "    if maze[new_r, new_c] == 1:\n",
    "        return state\n",
    "\n",
    "    return (new_r, new_c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "30bc4840-02b5-40b7-a3b3-48e436fc7cd8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training completed!\n"
     ]
    }
   ],
   "source": [
    "for episode in range(episodes):\n",
    "\n",
    "    state = start\n",
    "\n",
    "    for step in range(max_steps):\n",
    "\n",
    "        # Epsilon-greedy action selection\n",
    "        if random.random() < epsilon:\n",
    "            action = random.randint(0, 3)\n",
    "        else:\n",
    "            action = np.argmax(\n",
    "                q_table[state[0], state[1]]\n",
    "            )\n",
    "\n",
    "        # Get next state\n",
    "        next_state = get_next_state(state, action)\n",
    "\n",
    "        # Calculate reward\n",
    "        if next_state == goal:\n",
    "            reward = 100\n",
    "        elif next_state == state:\n",
    "            reward = -10\n",
    "        else:\n",
    "            reward = -1\n",
    "\n",
    "        # Q-learning update\n",
    "        old_value = q_table[\n",
    "            state[0], state[1], action\n",
    "        ]\n",
    "\n",
    "        next_max = np.max(\n",
    "            q_table[next_state[0], next_state[1]]\n",
    "        )\n",
    "\n",
    "        new_value = old_value + learning_rate * (\n",
    "            reward + discount_factor * next_max - old_value\n",
    "        )\n",
    "\n",
    "        q_table[\n",
    "            state[0], state[1], action\n",
    "        ] = new_value\n",
    "\n",
    "        state = next_state\n",
    "\n",
    "        # Stop when goal is reached\n",
    "        if state == goal:\n",
    "            break\n",
    "\n",
    "    # Reduce exploration\n",
    "    epsilon = max(epsilon_min, epsilon * epsilon_decay)\n",
    "\n",
    "print(\"Training completed!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "7d68bcf9-2945-4fdb-8302-5fda42c1382c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Q-Table:\n",
      "[[[ 35.84  35.84  35.84  48.26]\n",
      "  [ 39.26  39.26  44.84  51.85]\n",
      "  [ 42.85  55.63  48.26  42.85]\n",
      "  [  0.     0.     0.     0.  ]\n",
      "  [ 35.81  48.26  35.75  35.84]]\n",
      "\n",
      " [[  0.     0.     0.     0.  ]\n",
      "  [  0.     0.     0.     0.  ]\n",
      "  [ 51.85  59.61  46.63  46.63]\n",
      "  [  0.     0.     0.     0.  ]\n",
      "  [ 44.84  51.85  39.25  39.26]]\n",
      "\n",
      " [[ 59.21  72.85  59.21  63.8 ]\n",
      "  [ 54.8   54.8   68.21  59.61]\n",
      "  [ 55.63  50.61  63.8   55.63]\n",
      "  [ 46.63  46.63  59.61  51.85]\n",
      "  [ 48.26  42.85  55.63  42.85]]\n",
      "\n",
      " [[ 68.21  77.74  63.85  63.85]\n",
      "  [  0.     0.     0.     0.  ]\n",
      "  [  0.     0.     0.     0.  ]\n",
      "  [  0.     0.     0.     0.  ]\n",
      "  [  0.     0.     0.     0.  ]]\n",
      "\n",
      " [[ 72.85  68.74  68.74  82.88]\n",
      "  [ 73.88  73.88  77.74  88.3 ]\n",
      "  [ 79.3   79.3   82.88  94.  ]\n",
      "  [ 85.    85.    88.3  100.  ]\n",
      "  [  0.     0.     0.     0.  ]]]\n"
     ]
    }
   ],
   "source": [
    "print(\"Q-Table:\")\n",
    "print(np.round(q_table, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "2e602b86-795a-4642-98db-28d8912cf6ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Learned Path:\n",
      "[(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]\n",
      "\n",
      "Goal reached successfully!\n"
     ]
    }
   ],
   "source": [
    "state = start\n",
    "path = [state]\n",
    "\n",
    "for step in range(max_steps):\n",
    "\n",
    "    # Select best action\n",
    "    action = np.argmax(\n",
    "        q_table[state[0], state[1]]\n",
    "    )\n",
    "\n",
    "    next_state = get_next_state(state, action)\n",
    "\n",
    "    # Avoid infinite loop\n",
    "    if next_state == state:\n",
    "        break\n",
    "\n",
    "    state = next_state\n",
    "    path.append(state)\n",
    "\n",
    "    # Stop at goal\n",
    "    if state == goal:\n",
    "        break\n",
    "\n",
    "print(\"Learned Path:\")\n",
    "print(path)\n",
    "\n",
    "if state == goal:\n",
    "    print(\"\\nGoal reached successfully!\")\n",
    "else:\n",
    "    print(\"\\nGoal could not be reached.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "49455a7e-7b56-4205-8200-9126b6ffdda6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Maze with Learned Path:\n",
      "S * * 1 0\n",
      "1 1 * 1 0\n",
      "* * * 0 0\n",
      "* 1 1 1 1\n",
      "* * * * G\n"
     ]
    }
   ],
   "source": [
    "display_maze = maze.astype(str)\n",
    "\n",
    "for r, c in path:\n",
    "\n",
    "    if (r, c) == start:\n",
    "        display_maze[r, c] = \"S\"\n",
    "\n",
    "    elif (r, c) == goal:\n",
    "        display_maze[r, c] = \"G\"\n",
    "\n",
    "    else:\n",
    "        display_maze[r, c] = \"*\"\n",
    "\n",
    "print(\"Maze with Learned Path:\")\n",
    "\n",
    "for row in display_maze:\n",
    "    print(\" \".join(row))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f4a4c490-6723-4553-b471-556ec6d5ef1b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23242f9b-28c3-49ad-b055-41964d945868",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43e315fc-9e6a-4116-9d82-92200223e73d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "085fdbff-894c-4a4f-8e9e-18de2c1f5641",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f43b4282-d300-422c-b00a-f8e9a26e5a13",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88cff181-0f76-4802-aac1-0cab31fb7e10",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3f9735b-dd79-49eb-aeba-e4fe53ccbd95",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
