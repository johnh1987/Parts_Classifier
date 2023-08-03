from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump,load
import joblib



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
#print(unique_parts_list)

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
#print(unique_machine_list)


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
        if tdelta_s < 1000:
            # swap item text to index digit

            machine_identifier = str(unique_machine_list.index[unique_machine_list['Machine'] == mach_out].tolist()).strip('[]')
            pd_out['Seconds'].append(tdelta_s)
            pd_out['Machine'].append(machine_identifier)
    except:
        pass

df_data = pd.DataFrame(pd_out)

X_test = np.array(df_data)






# data cleanup for loops




#print(X_test)

y = model.predict(X_test)
#print(y)
x_to_df =str(X_test).split(']')
y_to_df = str(y).split()

test_dataset_df = {'Test Seconds':[],'Test Machine':[],'Predicted Result':[]}

special_characters = ['[', ']', "'",'\n']

for item in x_to_df:

    test_item_to_clean = item
    #print(test_item_to_clean)

    for clear in special_characters:
        test_item_to_clean = test_item_to_clean.replace(clear,"")
        test_item_to_clean_out = test_item_to_clean

    try:
        a = test_item_to_clean_out.split()

        #print("TEST A :",a[0])
        test_dataset_df['Test Seconds'].append(a[0])
        #print("TEST B :",a[1])
        test_dataset_df['Test Machine'].append(a[1])
    except:
        pass


for item in y_to_df:

    prediction_item_to_clean = item


    for clear in special_characters:
        prediction_item_to_clean = prediction_item_to_clean.replace(clear,"")
        prediction_item_to_clean_out = prediction_item_to_clean
    test_dataset_df['Predicted Result'].append(prediction_item_to_clean_out)



#print(test_dataset_df)
# convert to df and print prediction data
prediction_data = pd.DataFrame(test_dataset_df)
#print(prediction_data.head())

index_1 = 0
index_range_1 = (len(unique_parts_list))
index_2 = 0
index_range_2 = (len(unique_machine_list))

while index_1 != index_range_1:

    part_name = (unique_parts_list.iloc[int(index_1)]['Product'])
    #print(part_name)
    prediction_data['Predicted Result'] = prediction_data['Predicted Result'].replace(str(index_1),part_name)
    index_1 += 1



while index_2 != index_range_2:

    part_name = (unique_machine_list.iloc[int(index_2)]['Machine'])
    #print(part_name)
    prediction_data['Test Machine'] = prediction_data['Test Machine'].replace(str(index_2),part_name)
    index_2 += 1


print(type(prediction_data))
# output dataset dataframe


for index, row in prediction_data.iterrows():

    seconds = int(row['Test Seconds'])
    result = row['Predicted Result']
    #print(seconds)
    #print(result)

    if seconds < 30:
        #print("LESS THAN 30")
        out = 'NON CYCLE EVENT'
        prediction_data.at[index,'Predicted Result'] = out

    else:
        pass



print(prediction_data.to_string())




