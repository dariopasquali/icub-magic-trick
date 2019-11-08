import csv

with open("lie_features_real_time_heuristic__vad_all.csv", "r") as in_file:
    reader = csv.reader(in_file)
    keep_reading = True
    while(keep_reading):
        try:
            line = next(reader)
            print(line)
        except:
            keep_reading = False
        
        