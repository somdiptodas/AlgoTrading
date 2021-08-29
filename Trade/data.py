import openpyxl
from datetime import datetime

# my data rows as dictionary objects 
def saveData(data):
    mydict ={"A": data[0], 'B': data[1], 'C': data[2], 'D': data[3], 'E': datetime.now()}
    dt = str(datetime.now())
    # # field names 
        
    # name of csv file 
    filename = "orderInfo.xlsx"
    WB = openpyxl.load_workbook(filename)
    sheet = WB['Spread order data']
    sheet.append(mydict)
    
    WB.save(filename)

