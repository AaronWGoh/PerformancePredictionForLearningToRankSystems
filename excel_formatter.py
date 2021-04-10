from win32com.client import Dispatch

excel = Dispatch('Excel.Application')
wb = excel.Workbooks.Open("D:\\output.xlsx")

#Activate second sheet
excel.Worksheets(2).Activate()

#Autofit column in active sheet
excel.ActiveSheet.Columns.AutoFit()

#Save changes in a new file
wb.SaveAs("D:\\output_fit.xlsx")

#Or simply save changes in a current file
#wb.Save()

wb.Close()