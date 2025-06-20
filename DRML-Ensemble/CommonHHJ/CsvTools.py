import csv
import os
import argparse

def WriteData(filename, data_to_write):
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
        
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if data_to_write and isinstance(data_to_write, list):
            is_list_of_rows = bool(data_to_write) and isinstance(data_to_write[0], (list, tuple))

            if is_list_of_rows:
                for row in data_to_write:
                    writer.writerow(row if isinstance(row, (list, tuple)) else [row])
            else:
                for item in data_to_write:
                    if isinstance(item, argparse.Namespace):
                        writer.writerow(["Argument", "Value"])
                        for arg, value in vars(item).items():
                            writer.writerow([arg, value])
                        writer.writerow([])
                    elif isinstance(item, (list, tuple)):
                         writer.writerow(item)
                    else:
                        writer.writerow([str(item)])
        else:
            try:
                writer.writerow([str(data_to_write)])
            except:
                writer.writerow(["Could not serialize non-list data"])