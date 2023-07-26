import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import csv


df = pd.read_csv('TestData.csv', sep=',')

start_to_end = ""
State_check = 0
row_complete = 0

pd_out = {'Product':[],'Seconds':[] }

dfpl = []
current_list = pd.DataFrame(dfpl)


for index, row in df.iterrows():
    #print(index)

    if row['PRODUCT'] not in dfpl:

        dfpl.append(row['PRODUCT'])
    else:

        pass

parts_list = pd.DataFrame(dfpl,columns=['Product'])
print(parts_list,"\n")

for index, row in df.iterrows():


    #print(row['TIME'],row['STATE'],row['PRODUCT'])

    if row['STATE'] == 'CYCLE START ' and State_check == 0:

        start_to_end = ""
        row_out = "%s,%s," %(row['STATE'],row['TIME'])
        start_to_end += row_out
        State_check = 1
        row_complete = 0

    elif row['STATE'] == 'CYCLE END ' and State_check == 1:

        prod_out = row['PRODUCT']
        part_identifier = str(parts_list.index[parts_list['Product']==prod_out].tolist()).strip('[]')
        row_out = "%s,%s,%s" % (row['STATE'], row['TIME'],part_identifier)
        start_to_end += row_out
        State_check = 0
        row_complete = 1


    #print(start_to_end)
    if row_complete == 1:
        try:
            ste_split = start_to_end.split(',')

            t1 = (ste_split[1])
            t2 = (ste_split[3])
            product = (ste_split[4])

            FMT = '%H:%M:%S'
            tdelta = datetime.strptime(str(t2),FMT) - datetime.strptime(str(t1),FMT)
            string_con = tdelta.seconds

            if string_con <= 26:
                part_identifier = 5
            else:
                pass


            pd_out['Product'].append(int(part_identifier))
            pd_out['Seconds'].append((string_con))
            row_complete = 0


        except:
            #print("Error")
            pass

df2 = pd.DataFrame(pd_out)
df2.sort_values(by=['Seconds'],inplace=True)
print(df2.to_string())

y = np.array(df2['Product'])
x = np.array(df2['Seconds'])

scatter = px.scatter_matrix(df2)
scatter.show()

Knn = KNeighborsClassifier(n_neighbors=5)

x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1,1),y)
Knn.fit(x.reshape(-1,1),y)

special_characters=['[',']','\n']
normal_string = str(x_test)
for i in special_characters:
    normal_string = normal_string.replace(i,"")

pre = str(Knn.predict(x_test))
pre = str(pre.split(','))

test_x_data = (normal_string.replace('   ',' '))
prediction = ((Knn.predict(x_test)))
model_score = ("MODEL SCORE : ",round(100*(Knn.score(x.reshape(-1,1),y)),2),"%")

test_x_data = str(test_x_data).split()
prediction = str(prediction).split(' ')


#print(test_x_data)
#print(prediction)
#print(model_score)
file_path = "Prediction record.csv"



n = (test_x_data)

p = (prediction)
    #print(str(pred_test))

dict = {'Product prediction':[],'Time (s)':[]}



for j in n:
   if j.isdigit():
        #print(i)
        dict['Time (s)'].append(j)


for i in p:
    if i.isdigit():
#        #print(j)
         dict['Product prediction'].append(i)


print(dict)









