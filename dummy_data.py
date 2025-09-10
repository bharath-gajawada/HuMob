import csv
import random

filename = "data/city_F_challengedata.csv"
num_rows = 500  # Adjust for desired size

with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["uid", "d", "t", "x", "y"])
    for i in range(num_rows):
        uid = random.randint(1, 25)
        d = random.randint(1, 75)
        t = random.randint(1, 48)
        x = random.randint(1, 100)
        y = random.randint(1, 100)
        writer.writerow([uid, d, t, x, y])