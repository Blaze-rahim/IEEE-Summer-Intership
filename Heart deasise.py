import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tkinter import *
from tkinter import messagebox


def cardiotrainer(e):
    cardio_data = pd.read_csv('heart.csv')

    # Splitting the Target Column and storing in X
    x = cardio_data.drop(columns='target', axis=1)  # axis is for columns

    # Storing target data in Y
    y = cardio_data['target']

    # Training and testing X and Y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, stratify=y, random_state=2)

    # Training the ML using Logistic regression
    model = LogisticRegression(max_iter=2000)
    print(model.fit(x_train, y_train))

    x_train_prediction = model.predict(x_train)
    data_accuracy = accuracy_score(x_train_prediction, y_train)
    print(f"Accuracy is : {data_accuracy}")

    x_test_prediction = model.predict(x_test)
    data_accuracy = accuracy_score(x_test_prediction, y_test)
    print(f"Accuracy is : {data_accuracy}")

    cardioprediction(model, e)
    return 0


def cardioprediction(model, e):
    a = tuplemaker(e, input_values)
    data = np.asarray(a)
    resh_data = data.reshape(1, -1)
    prediction = model.predict(resh_data)

    if prediction[0] == 0:
        messagebox.showinfo("Result", "This person have no heart Disease")

    else:
        messagebox.showwarning('Warning', "This person have heart Disease, Please contact your doctor imediately")

    return 0


def makeform(root, fields):
    entries = {}
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=50, text=field + ": ", anchor='w')
        ent = Entry(row)
        ent.insert(0, "0")
        row.pack(side=TOP, fill=X, padx=20, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent

    return entries


def tuplemaker(entries, input_values):
    lst = []
    for i in list(input_values):
        var = float(entries[i].get())
        lst.append(var)

    return (lst)


# TKinter parts
tk = Tk()

input_values = ('Age', 'Sex(1=Male, 0=Women)', 'Chest Pain(0-3)', 'Resting bp', 'Cholestoral',
                'Fasting blood sugar(True = 1, False = 0)', 'Electrocardiographic(0-2)', 'Max heartrate',
                'Angina(1=True, 0 = False)', 'Old peak', 'Slope(0-2)', 'Major Vessels count(0-3)',
                'Thal(1 =Normal, 2 = Fixed defect, 3 = Reversable defect')

ents = makeform(tk, input_values)

b1 = Button(tk, text='Submit',
            command=(lambda e=ents: cardiotrainer(e)))
b1.pack(side=BOTTOM, padx=5, pady=5, )
tk.mainloop()
