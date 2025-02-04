import pandas as pd

data1 = pd.read_excel('data/Movement1.xlsx',
                      sheet_name='Sheet1', engine='openpyxl')
data1['Movement'] = 1
data2 = pd.read_excel('data/Movement1.xlsx',
                      sheet_name='Sheet2', engine='openpyxl')
data2['Movement'] = 2
data3 = pd.read_excel('data/Movement1.xlsx',
                      sheet_name='Sheet3', engine='openpyxl')
data3['Movement'] = 3
print(data1)
print(data2)
print(data3)

final_data = data1._append(data2, ignore_index=True)
final_data = final_data._append(data3, ignore_index=True)
print(final_data)

print(final_data.info())
final_data.to_csv('data/final_data.csv', index=False)
