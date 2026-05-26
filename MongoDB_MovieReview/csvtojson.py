import csv
import json

# Input and output file paths
csv_file = "C:\\Users\\slpmi\\OneDrive\\Desktop\\contus\\MovieReviewMongodb\\rotten_tomatoes_critic_reviews.csv"
json_file = "MovieReviewData.json"

# Read CSV and convert to JSON
data = []

with open(csv_file, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)

    for row in csv_reader:
        data.append(row)

# Write JSON data to file
with open(json_file, mode='w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)

print(f"CSV data has been converted to JSON and saved in '{json_file}'")