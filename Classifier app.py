from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump,load
import joblib
import pickle

model_save = "ML KNN Model V.1.pickle"

model = joblib.load(model_save)


df = pd.read_csv('July_Data.csv', sep=',')
#print(df)
# unique product list
upl = []

# find unique items in products
for index, row in df.iterrows():
    #print(index)

    if row['Comment'] not in upl:

        upl.append(row['Comment'])
    else:

        pass


upl.append('Non Cycle')
# convert upl to dataframe
unique_parts_list = pd.DataFrame(upl,columns=['Product'])
print(unique_parts_list)

# unique machine list
uml = []

for index, row in df.iterrows():
    #print(index)

    if row['Machine'] not in uml:

        uml.append(row['Machine'])
    else:

        pass

# convert uml to dataframe
unique_machine_list = pd.DataFrame(uml,columns=['Machine'])
print(unique_machine_list)


pd_out = {'Seconds':[] , 'Machine':[]}

# convert csv to pandas dataframe
df2 = pd.read_csv('d.csv', sep=',')

#print(df)

for index, row in df2.iterrows():


    start_t = row['Start_time']
    end_t = row['End_time']
    FMT = '%H:%M:%S'


    try:
        tdelta = datetime.strptime(str(end_t), FMT) - datetime.strptime(str(start_t), FMT)
        tdelta_s = tdelta.seconds
        mach_out = (row['Machine'])

        # remove outlier values including non cycle events
        if 30 < tdelta_s < 1000:
            # swap item text to index digit

            machine_identifier = str(unique_machine_list.index[unique_machine_list['Machine'] == mach_out].tolist()).strip('[]')
            pd_out['Seconds'].append(tdelta_s)
            pd_out['Machine'].append(machine_identifier)
    except:
        pass

df_data = pd.DataFrame(pd_out)

X_test = np.array(df_data)






# data cleanup for loops




print(X_test)

y = model.predict(X_test)
print(y)




