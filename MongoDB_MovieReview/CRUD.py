from pymongo import MongoClient

# Connect with Mongo Database 
client = MongoClient("mongodb://localhost:27017/")
db = client["MovieReviewDB"]
collection = db["Reviews"]

# CRUD Operations

# # Read Operations
# print("\nAll Reviews:\n")

# for review in collection.find():
#     print(review)

# Filtering

print("\Content with publiser name = Sunday Mail\n")

query = {"publisher_name": "Sunday Mail (Australia)"}

projection = {"_id": 0,"rotten_tomatoes_link": 1,"publisher_name": 1,"review_type": 1,"review_date": 1,"review_content": 1}
reviews = collection.find(query, projection).limit(10)
for review in reviews:
    print(review)

# Limiting
query = {"publisher_name": "Sunday Mail (Australia)"}
reviews = collection.find(query).limit(5)

for review in reviews:
    print(review)

# Sorting
query = {"publisher_name": "Sunday Mail (Australia)"}
reviews = collection.find(query).sort("review_date", 1)
for review in reviews:
    print(review)
    
# Update Database
collection.update_one({"movie": "Interstellar"},{"$set": {"rating": 5,"review": "Masterpiece movie"}})
print("\nReview Updated Successfully\n")

# Delete operation
collection.delete_one({"critic_name": "Andrew L. Urban"})
print("Review Deleted Successfully")

# Join Operation
reviews_collection = db["Reviews"]
x = [
    {
        "$match": {"publisher_name": "Sunday Mail (Australia)"}
    },
    {
        "$lookup": {"from": "Movies","localField": "rotten_tomatoes_link","foreignField": "rotten_tomatoes_link","as": "movie_details"}
    },
    {
        "$unwind": "$movie_details"
    },
    {
        "$sort": {"review_date": -1}
    },
    {
        "$limit": 5
    },
    {
        "$project": {"_id": 0,"critic_name": 1,"publisher_name": 1,"review_type": 1,"review_score": 1,"review_date": 1,"review_content": 1,"movie_details.movie_name": 1,"movie_details.genre": 1,"movie_details.year": 1}
    }
]
results = reviews_collection.aggregate(x)
print("Joined Movies collections with Review Collections")
for review in results:
    print("Movie Name   :", review["movie_details"]["movie_name"])
    print("Genre        :", review["movie_details"]["genre"])
    print("Year         :", review["movie_details"]["year"])
    print("Critic       :", review["critic_name"])
    print("Publisher    :", review["publisher_name"])
    print("Review Type  :", review["review_type"])
    print("Score        :", review["review_score"])
    print("Review Date  :", review["review_date"])
    print("Review       :", review["review_content"])

