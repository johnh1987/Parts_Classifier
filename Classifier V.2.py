import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

pd_out = {'Product':[],'Seconds':[],'Machines':[] }

# convert csv to pandas dataframe
df = pd.read_csv('July_Data.csv', sep=',')

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

tp = {'Product':[],'Seconds':[], 'Machine':[]}

# find start and end times and calculate elapsed time in seconds
for index, row in df.iterrows():


    start_t = row['Start_time']
    end_t = row['End_time']
    FMT = '%H:%M:%S'

    try:
        tdelta = datetime.strptime(str(end_t), FMT) - datetime.strptime(str(start_t), FMT)
        tdelta_s = tdelta.seconds


        # remove outlier values including non cycle events
        if 30 < tdelta_s < 1000:

            prod_out = (row['Comment'])
            mach_out = (row['Machine'])

            # swap item text to index digit
            part_identifier = str(unique_parts_list.index[unique_parts_list['Product'] == prod_out].tolist()).strip('[]')
            machine_identifier = str(unique_machine_list.index[unique_machine_list['Machine'] == mach_out].tolist()).strip('[]')

            # append tp dictionary
            tp['Product'].append(part_identifier)
            tp['Seconds'].append(tdelta_s)
            tp['Machine'].append(machine_identifier)

        else:
            pass

    except:
        pass

#convert tp dictionary to data frame
testing_data = pd.DataFrame(tp)
testing_data.sort_values(by=['Seconds'],inplace=True)
#print(testing_data.to_string())

# knn axis assignment
y = np.array(testing_data['Product'])
x = np.array(testing_data.drop('Product',axis=1))

#print(y)
#print(x)

# train data setup with test size split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.1)


# determine optimum k value
model_score_list = []

for k in range(1,40):

    # amount of nearest neighbours required
    knn = KNeighborsClassifier(n_neighbors=k,algorithm='ball_tree',weights='distance')

    # train algorithm on training data
    knn.fit(X_train,y_train)

    # categories new data based on trained dataset
    prediction_k_test = knn.predict(X_test)

    #print("X TEST ",X_test)
    #print("Y TEST ",y_test)

    # output model score
    model_score = round(100*(knn.score(x,y)),2)
    #print("K-Value : ",k," Model Score",model_score,"%")
    model_score_list.append(model_score)


print((max(model_score_list)),"%")

k = model_score_list.index(max(model_score_list)) + 1
#print(k)

# amount of nearest neighbours required
knn = KNeighborsClassifier(n_neighbors=k,algorithm='ball_tree',weights='distance')

# train algorithm on training data
knn.fit(X_train,y_train)

# categories new data based on trained dataset
prediction_test = knn.predict(X_test)

# data clean part
x_to_df = str(X_test).split(']')
y_to_df = str(y_test).split()

# dictionary to append values to before turning into data frame
test_dataset_df = {'Test Seconds':[],'Test Machine':[],'Predicted Result':[]}

# data cleanup for loops
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

    #print("Prediction : ",prediction_item_to_clean_out)
    test_dataset_df['Predicted Result'].append(prediction_item_to_clean_out)


# convert to df and print prediction data
prediction_data = pd.DataFrame(test_dataset_df)
#print(prediction_data.head())

index_1 = 0
index_range_1 = (len(unique_parts_list))

while index_1 != index_range_1:

    part_name = (unique_parts_list.iloc[int(index_1)]['Product'])
    #print(part_name)
    prediction_data['Predicted Result'] = prediction_data['Predicted Result'].replace(str(index_1),part_name)
    index_1 += 1

index_2 = 0
index_range_2 = (len(unique_machine_list))

while index_2 != index_range_2:

    part_name = (unique_machine_list.iloc[int(index_2)]['Machine'])
    #print(part_name)
    prediction_data['Test Machine'] = prediction_data['Test Machine'].replace(str(index_2),part_name)
    index_2 += 1


# output dataset dataframe
prediction_data.sort_values(by=['Predicted Result'],inplace=True)
print(prediction_data.to_string())
#prediction_data.to_csv('KNN Test data')
