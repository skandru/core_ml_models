import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Sample email data
emails = [
    # Spam examples
    "URGENT: You've won $1,000,000! Claim now!",
    "Buy cheap medications online! Best deals!",
    "Work from home and earn $5000/week guaranteed",
    "CONGRATULATIONS! You're our lucky winner",
    "Increase your followers instantly! Buy now!",
    "Amazing weight loss pill, lose 10kg in 1 week",
    "FREE iPhone 14 Pro - Click here to claim",
    "Your account has been suspended! Verify now",
    "Meet singles in your area tonight!",
    "Double your money in just 24 hours - invest now",
    
    # Non-spam examples
    "Meeting scheduled for tomorrow at 10 AM",
    "Please review the attached project proposal",
    "Your Amazon order has been shipped",
    "Thank you for your recent purchase",
    "Flight confirmation: NYC to LAX",
    "Your monthly account statement is ready",
    "Team lunch next Wednesday at 12:30",
    "Reminder: Dentist appointment tomorrow",
    "Updates to our privacy policy",
    "Your subscription has been renewed"
]

# Labels (1 for spam, 0 for non-spam)
labels = [1] * 10 + [0] * 10  # First 10 are spam, last 10 are non-spam

print("Step 1: Data Split")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.3, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nStep 2: Feature Extraction")
# Create TF-IDF vectorizer with adjusted parameters
vectorizer = TfidfVectorizer(
    min_df=1,  # Changed from 2 to 1 to include all terms
    stop_words='english',
    lowercase=True,
    max_features=None  # Remove limit on features
)

# Transform the text data into TF-IDF features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Number of features: {len(vectorizer.get_feature_names_out())}")

print("\nStep 3: Model Training")
# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)
print("Model training completed")

print("\nStep 4: Model Evaluation")
# Make predictions on test set
y_pred = classifier.predict(X_test_tfidf)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Spam', 'Spam']))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("Format: [[TN, FP], [FN, TP]]")
print(cm)

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_train_tfidf, y_train, cv=3)  # Changed to 3-fold CV due to small dataset
print("\nCross-validation scores:", cv_scores)
print("Average CV score: {:.2f}".format(cv_scores.mean()))

# Function to classify new emails
def classify_email(email_text):
    # Transform the new email using the same vectorizer
    email_tfidf = vectorizer.transform([email_text])
    
    # Get the prediction
    prediction = classifier.predict(email_tfidf)[0]
    
    # Get probability scores
    prob_scores = classifier.predict_proba(email_tfidf)[0]
    
    return {
        'classification': 'Spam' if prediction == 1 else 'Non-Spam',
        'confidence': prob_scores[prediction]
    }

print("\nStep 5: Testing New Emails")
# Test the classifier with new examples
new_emails = [
    "URGENT: Your account needs verification!",
    "Meeting rescheduled to 3 PM tomorrow",
    "WIN WIN WIN! Lottery results inside"
]

print("\nTesting new emails:")
for email in new_emails:
    result = classify_email(email)
    print(f"\nEmail: {email}")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2f}")

print("\nStep 6: Feature Importance Analysis")
# Feature importance analysis
feature_names = vectorizer.get_feature_names_out()
feature_importances = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
top_features = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 spam-indicating words:")
print(top_features.head(10)[['feature', 'importance']].to_string())
print("\nTop 10 non-spam-indicating words:")
print(top_features.tail(10)[['feature', 'importance']].to_string())