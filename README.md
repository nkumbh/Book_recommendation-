# Book_recommendation-
\
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer


# In[3]:


# Load dataset using Pandas:

encodings = ['utf-8', 'ISO-8859-1', 'latin1']  # List of encodings to try

for encoding in encodings:
    try:
        book_data = pd.read_csv("book.csv", encoding=encoding)
        print("File read successfully with encoding:", encoding)
        break  # Stop trying encodings if successful
    except UnicodeDecodeError:
        print("Error reading file with encoding:", encoding)


# In[4]:


book_data


# In[6]:


book_data.info()


# In[12]:


book_data = book_data.drop(book_data.columns[0], axis=1)


# In[13]:


book_data


# In[14]:


# Step 1: Handle missing values
book_data.fillna(0, inplace=True)  # Fill missing values with 0 for ratings


# In[15]:


# Step 2: Analyze data distribution
print("Data Distribution:")
print(book_data.describe())


# In[18]:


# Step 3: Generate rating matrix using "User.ID" and "Book.Title"
rating_matrix = book_data.pivot_table(index='User_ID', columns='Book_Title', values='Book_Rating', fill_value=0)


# In[19]:


rating_matrix


# In[20]:


# Step 4: Find the most similar user
def find_most_similar_user(user_id, rating_matrix):
    user_ratings = rating_matrix.loc[user_id].values.reshape(1, -1)
    similarities = cosine_similarity(rating_matrix.values, user_ratings)
    similar_user_index = similarities.argmax()
    return rating_matrix.index[similar_user_index]


# In[21]:


# Step 5: Perform recommendation
def recommend_books(user_id, rating_matrix, num_recommendations=5):
    similar_user_id = find_most_similar_user(user_id, rating_matrix)
    user_ratings = rating_matrix.loc[similar_user_id]
    user_ratings_sorted = user_ratings.sort_values(ascending=False)
    recommended_books = user_ratings_sorted.index[:num_recommendations]
    return recommended_books


# In[29]:


# Example usage

user_id = input("Enter User ID (from 8 to 278854): ")


# In[30]:


try:
    
    user_id = int(user_id) # convert input to integer
    recommended_books = recommend_books(user_id, rating_matrix)
    print("Recommended Books for User ID", user_id, ":")
    print(recommended_books)
except ValueError:
    print("User ID does not exists.")
    if user_id not in rating_matrix.index:
        print("User ID does not exist.")


# In[ ]:





# In[ ]:




