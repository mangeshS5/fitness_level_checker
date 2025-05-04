from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Sample dataset
data = {
    'Age': [25, 32, 28, 45, 30, 35, 40, 50, 22, 27],
    'BMI': [22.5, 28.3, 24.0, 30.5, 26.4, 23.0, 27.8, 29.9, 21.0, 25.2],
    'ExerciseFreq': [4, 1, 3, 0, 2, 4, 1, 0, 5, 3],
    'HeartRate': [65, 85, 70, 90, 78, 66, 82, 88, 60, 72],
    'FitnessStatus': ['Fit', 'Not Fit', 'Fit', 'Not Fit', 'Fit',
                      'Fit', 'Not Fit', 'Not Fit', 'Fit', 'Fit']
}

df = pd.DataFrame(data)
le = LabelEncoder()
df['FitnessEncoded'] = le.fit_transform(df['FitnessStatus'])

X = df[['Age', 'BMI', 'ExerciseFreq', 'HeartRate']]
y = df['FitnessEncoded']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        exercise = float(request.form['exercise'])
        heart_rate = float(request.form['heart_rate'])

        prediction = model.predict([[age, bmi, exercise, heart_rate]])[0]
        result = le.inverse_transform([prediction])[0]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
