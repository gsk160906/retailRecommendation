# ============================================
# RETAIL RECOMMENDATION ENGINE
# ============================================

# Import libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1 : Load Dataset
# -------------------------------
data = pd.read_csv("retail_data.csv")

print("\nOriginal Dataset\n")
print(data)

# -------------------------------
# Step 2 : Create Customer-Product Matrix
# -------------------------------
customer_product = data.pivot_table(
    index='Customer',
    columns='Product',
    values='Rating',
    fill_value=0
)

print("\nCustomer Product Matrix\n")
print(customer_product)

# -------------------------------
# Step 3 : Calculate Similarity
# -------------------------------
similarity = cosine_similarity(customer_product)

similarity_df = pd.DataFrame(
    similarity,
    index=customer_product.index,
    columns=customer_product.index
)

print("\nCustomer Similarity Matrix\n")
print(similarity_df)

# -------------------------------
# Step 4 : Recommendation Function
# -------------------------------
def recommend_products(customer_name):

    print("\nRecommendations for", customer_name)

    # Most similar customer
    similar_customer = similarity_df[customer_name].sort_values(ascending=False)

    # Ignore the customer itself
    similar_customer = similar_customer.drop(customer_name)

    best_match = similar_customer.idxmax()

    print("Most Similar Customer:", best_match)

    # Products purchased by customer
    customer_products = customer_product.loc[customer_name]

    # Products purchased by similar customer
    match_products = customer_product.loc[best_match]

    recommendations = []

    for product in customer_product.columns:

        if customer_products[product] == 0 and match_products[product] > 0:
            recommendations.append(product)

    if recommendations:
        print("Recommended Products:")
        for item in recommendations:
            print("-", item)
    else:
        print("No new recommendations available.")

# -------------------------------
# Step 5 : Test
# -------------------------------
recommend_products("Alice")
recommend_products("David")
recommend_products("Emma")