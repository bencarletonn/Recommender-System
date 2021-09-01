import pandas as pd 
import os 
import numpy as np 
from surprise import Reader, Dataset, SVD 

# Loading the datasets 
print("Loading datasets...") 
PATH1 = os.getcwd() + "/reviewswithouttext.pkl" # load as .pkl as we removed 'text' column
PATH2 = os.getcwd() + "/yelp_academic_dataset_business.json"
PATH3 = os.getcwd() + "/yelp_academic_dataset_covid_features.json" 
df1 = pd.read_pickle(PATH1) 
df2 = pd.read_json(PATH2, lines=True)
df3 = pd.read_json(PATH3, lines=True)

print("Undergoing data preperation and feature selection...") 
# Editing reviews dataset
df_reviews = df1.drop(['useful', 'funny', 'cool'], axis=1) # Remove unneccesary columns

# Editing business datset
df_business = df2.drop(['address', 'city', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'attributes', 'hours'], axis=1) # Remove unneccesary columns 
df_business = df_business[df_business.is_open != 0] # Drop businesses that are no longer open
df_business = df_business.dropna(axis=0) # Remove rows with missing data
df_business = df_business.reset_index(drop=True) # Reset index 
category = "Restaurants" # Filter businesses on desired category, i.e. restaurants 
df_business = df_business.loc[lambda df_business: df_business.categories.apply(lambda l: category in l)] # Removes all non-restaurant data 

# Editing COVID dataset
df3 = df3.drop(["highlights", "Covid Banner", "Temporary Closed Until", "Virtual Services Offered"], axis=1) # Remove ambigious features

# Merge business and reviews 
df_business_review = pd.merge(df_business, df_reviews)  
year = "2019" # Filter so only contains reviews from 2019
df_business_review = df_business_review.loc[lambda df: df_business_review.date.apply(lambda l: year in l)] # Removes all non-2019 data 

# Remove all columns that contains a user with <15 reviews 
value_counts = df_business_review['user_id'].value_counts()
df_val_counts = pd.DataFrame(value_counts)
df_user = df_val_counts.reset_index()
df_user.columns = ['user_id', 'total_reviews']
# Remove users with less than 15 reviews 
df_user = df_user[df_user["total_reviews"] >= 15]
# Merge business, reviews and user so we get table containing user reviews of restaurants, where each user has atleast 15+ reviews
df_master = pd.merge(df_business_review, df_user) 
content_df = df_master.copy(deep=True) # Taking a copy of this dataset to use for content-based filter
df_master = df_master.drop(["is_open", "categories", "state"], axis=1)
df_master = df_master.drop(['business_id', 'review_id', 'date', 'total_reviews'], axis=1) # df_master contains columns [name, user_id, stars], i.e. each users reviews 


# Create cosine similarity matrix for semantic-aware filtering 
content_df = content_df.drop(["is_open", "review_id", "stars", "user_id", "date", "total_reviews"], axis=1) # Removing unneccesary columns 
content_df = content_df.drop_duplicates(subset="name") 
content_df = content_df.reset_index().drop(["index"], axis=1)
# Convert categories from str to list 
content_df['categories'] = content_df.categories.apply(lambda x: x.split(', '))
content_copy = content_df.copy() 

# function converts list of categories to additional columns
# based off https://stackoverflow.com/questions/37474001/converting-list-in-panda-dataframe-into-columns

# List of categories we want to include 
new_categories = ['Persian/Iranian','Turkish', 'Middle Eastern','Kebab','Steakhouses','Seafood', 'Modern European','Russian', 'Mediterranean','Sushi Bars', 'Japanese','Italian', 'Mexican', 'Cocktail Bars', 'Bars', 'Pubs', 'American (Traditional)','Tapas/Small Plates','Coffee & Tea','Thai', 'Desserts', 'Salad', 'Sandwiches','Breakfast & Brunch','Barbeque', 'Spanish', 'Indian', 'Pizza', 'Vietnamese','Malaysian','Greek', 'Burgers', 'Fast Food','Vegetarian','Gastropubs','Caribbean','Portuguese','Sports Bars']
def list2columns(df):
    df = df.copy()
    columns=['categories']
    for col in columns:
        for i in range(len(df)):
            for category in df.loc[i,col]:
                if category in new_categories:
                    if category not in df.columns:
                        df.loc[:,category] = 0
                        df[category][i]  = 1 
                    else:
                        df[category][i]  = 1
    return df

content_df = list2columns(content_df) 
# drop categories column 
content_df = content_df.drop(["categories", "business_id", "state"], axis=1)
# KNN approach using cosine similarity on semantic features
from sklearn.metrics.pairwise import cosine_similarity
content_matrix = content_df.drop(['name'], axis=1)
cosine_sim_matrix = cosine_similarity(content_matrix, content_matrix)


# Create functions for semantic-aware and colab filtering 
def content(already_reviewed, df_user_ratings, df_master, content_df, cosine_sim_matrix): 

	content_ratings = {} 
	review_counter = 0 

	for i in range(len(already_reviewed.index)):
		if already_reviewed[i] >= 4: # If positive review of restaurant 
			review_counter += 1 
			restaurant_index = (content_df.name[content_df.name == already_reviewed.index[i]].index.tolist())[0]
			content_predictions = cosine_sim_matrix[restaurant_index]
			for index, rating in enumerate(content_predictions):
				index = content_df.name[index] 
				if index in content_ratings:
					content_ratings[index] = content_ratings[index] + rating 
				else:
					content_ratings[index] = rating 

	# Normalize ratings 
	content_ratings = {k: v / review_counter for k, v in content_ratings.items()}  # review_counter can be adjusted to account for different weighting, i.e smaller gives more weight to content
	max_sim = content_ratings[max(content_ratings, key=content_ratings.get)] # Divide by the element with max similarity, assume max similarity has rating of 5 star 
	content_ratings = {k: (v / max_sim) * 5 for k, v in content_ratings.items()}
	content_predictions = pd.Series(content_ratings, name="content_score").sort_values(ascending=False)
	content_predictions = content_predictions.drop(already_reviewed.index)
	content_predictions = pd.DataFrame({'name': content_predictions.index, 'rating' : content_predictions.values})
	return content_predictions 



def colab(user, df_user_ratings, df_master):
	# Define the format of the data 
	reader = Reader(rating_scale=(1,5)) 
	# Load the data from the pandas dataframe 
	data = Dataset.load_from_df(df_master[["user_id", "name", "stars"]], reader)

	model = SVD(n_factors=50, n_epochs=10, lr_all=0.005, reg_all=0.4) # Hypertuned parameters using gridsearch
	model.fit(data.build_full_trainset()) 

	user_name = df_user_ratings.iloc[user].name
	# Colab predictions 
	A = list(df_master['name'].unique()) 
	B = sorted(list(set(list(df_master[df_master['user_id'] == user_name]['name']))))
	predictions = [x for x in A if x not in B] # Restaurants user has not reviewed
	prediction_df = pd.DataFrame(columns=['name', 'rating']) 

	d = {} 

	for restaurant in predictions:
	  pred = model.predict(user_name, restaurant) 
	  d[restaurant] = pred.est 
	prediction_df = pd.DataFrame.from_dict(d, orient='index') # columns=['name', 'rating']
	prediction_df = prediction_df.rename_axis(['name']).reset_index()
	prediction_df = prediction_df.rename(columns={0: "rating"})
	colab_prediction = prediction_df.sort_values(ascending=False, by="rating")
	colab_prediction = (colab_prediction.reset_index()).drop(columns=['index'], axis=1)

	
	return colab_prediction



def hybrid(content_predictions, colab_predictions, df2, df3, num_recommendations, user_index):
	finalRatings = content_predictions.merge(colab_predictions, on='name')
	finalRatings['Rating'] = ((finalRatings.rating_x + finalRatings.rating_y) / 2)
	finalRatings = finalRatings.drop(['rating_x', 'rating_y'], axis=1)
	finalRatings = finalRatings.sort_values(ascending=False, by='Rating').reset_index().drop(columns=['index'], axis=1)

	# Add option to filter based on COVID guidelines + state 
	business_info = df2.drop(['address', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours'], axis=1)
	business_info = pd.merge(business_info, df3, on="business_id").drop(['business_id'], axis=1)
	userOutput = finalRatings 
	userOutput = pd.merge(userOutput, business_info, on="name")
	userOutput = userOutput.groupby('name').agg({'Rating':'first', 'city': ', '.join, 'state': ', '.join, 'delivery or takeout' : 'first', 'Grubhub enabled' : 'first', 'Call To Action enabled' : 'first', 'Request a Quote Enabled' : 'first'}).reset_index()
	userOutput = userOutput.sort_values(ascending=False, by="Rating")

	# Filter on state 
	print("\nFilter by location\n\nPlease enter a location using the key below:\n\nAZ = Arizona\t\tON = Ontario\tPA = Pennsylvania\tNV = Nevada\nNC = North Carolina\tQC = Quebec\tOH = Ohio\t\tAB = Alberta\nWI = Wisconsin\t\tIL = Illinois\tSC = South Carolina\n\nOr enter 'NO' to not filter by state.")
	state = input(": ")
	if state == 'NO':
		pass
	else: # If they have entered a state 
		userOutput = userOutput[userOutput['state'].str.contains(state)]

	# Filter on COVID 
	print("Please enter 'YES' or 'NO' in response to the following questions regarding COVID-19 regulations\nNOTE: This data is not stored and serves only to filter the recommendations to ensure safety.")


	print("Does the restaurant have to deliver?") 
	deliver = input(": ") 
	if deliver == "YES":
		userOutput = userOutput[userOutput['delivery or takeout'] == 'TRUE']

	print("Does the restaurant have to be available on Grubhub?")
	grubhub = input(": ") 
	if grubhub == "YES":
		userOutput = userOutput[userOutput['Grubhub enabled'] == 'TRUE'] 
 

	userOutput = userOutput.reset_index().drop(columns=["index", "city", "delivery or takeout", "Grubhub enabled", "Call To Action enabled", "Request a Quote Enabled"], axis=1) 
	# Make sure state information looks nice 
	userOutput['state'] = userOutput.state.apply(lambda x: x.split(', '))
	userOutput['state'] = userOutput.state.apply(lambda x: list(set(x))) 

	string = f"\nHere are user {user_index}'s' 10 top recommendations:\n\n" 
	counter = 1 
	for i in range(len(userOutput.head(num_recommendations))):
	  name, rating, city = userOutput.iloc[i]
	  city = ', '.join(city) 
	  string += f"\t{counter}.\t{name}, located in {city}\n"
	  counter += 1 

	string += "\nThese recommendations were made by finding similar restaurants to those you have rated highly, as-well as identifying underlying features in restaurants that you like."
	return string 


# Initial
df_user_ratings = df_master.pivot_table(index="user_id", columns="name", values="stars").fillna(0)

# Command line interface 
print("\n\n\n\n\n\n\n\n\n\n\t\tHybrid Recommender System\n\n")
while True:
	print("\n\n\nEnter 'R' to view user recommendations or 'E' to edit a user profile:\n")
	command = input(": ") 

	if command == 'R': # V

		m = len(df_user_ratings.index)-1
		print(f"\n\nPlease enter a user index (from 0-{m}):\n") 
		user = int(input(": "))
		already_reviewed = df_user_ratings.iloc[user] 
		already_reviewed = already_reviewed[already_reviewed != 0] 
		# Print already_reviewed is nice format 
		already_reviewed_string = f"\n User {user}'s previous reviews:\n"
		for i in range(len(already_reviewed.index)):
			already_reviewed_string += f"\t{already_reviewed.index[i]} - {int(already_reviewed.values[i])} stars\n"

		print(already_reviewed_string)

		content_predictions = content(already_reviewed, df_user_ratings, df_master, content_df, cosine_sim_matrix) 
		colab_predictions = colab(user, df_user_ratings, df_master) 
		hybrid_predictions = hybrid(content_predictions, colab_predictions, df2, df3, 10, user) 
		print(hybrid_predictions)





	elif command == 'E':
		m = len(df_user_ratings.index)-1
		print(f"\n\nPlease enter the user index to who you wish to edit (from 0-{m}):\n")
		user = int(input(": "))
		already_reviewed = df_user_ratings.iloc[user] 
		already_reviewed = already_reviewed[already_reviewed != 0] 
		# Print already_reviewed is nice format 
		already_reviewed_string = f"\n User {user}'s previous reviews:\n"
		for i in range(len(already_reviewed.index)):
			already_reviewed_string += f"\t{already_reviewed.index[i]} - {int(already_reviewed.values[i])} stars\n"

		print(already_reviewed_string)

		while True:
			print("Enter the name of the restaurant you would like to review (CaSe SeNsItIvE):")
			print("If you are finished entering new reviews. Type DONE")
			restaurant = input(": ") 
			if restaurant == "DONE":
				break
			if restaurant in df_master.name.unique(): # if restaurant exists
				print("Please rate the restaurant out of 5 stars") 
				stars = float(input(": "))
				if stars > 5:
					stars = 5 
				if stars <= 0:
					stars = 1

				userid = df_user_ratings.iloc[user].name 
				new_row = pd.DataFrame(columns=['name', 'user_id', 'stars'])
				row = {'name': restaurant, 'user_id': userid, 'stars': stars}
				new_row = new_row.append(row, ignore_index=True) 
				df_master = pd.concat([df_master, new_row], ignore_index=True)
			else:
				print("Restaurant doesn't exist. Enter a new restaurant")

		print("Recreating user-item matrix...") 
		df_user_ratings = df_master.pivot_table(index="user_id", columns="name", values="stars").fillna(0) 


	else:
		print("Command not found. Please try again") 