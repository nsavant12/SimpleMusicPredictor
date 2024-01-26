import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x, y)

age = input("Enter your age:")
g = input("Enter your gender (1=male,0=female):")
print(age)
print(g)

predictions = model.predict([ [age, g] ])
predictions
