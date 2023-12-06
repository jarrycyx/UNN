import csv

def read_csv(item_sel_path, encoding="utf-8"):
    # item_sel_path = "data_prep/files/items_sel2.csv"
    f = open(item_sel_path, "r", encoding=encoding)
    reader = csv.reader(f)
    items_sel = []
    for i, row in enumerate(reader):
        # print(row)
        items_sel.append(row[0])
    return items_sel