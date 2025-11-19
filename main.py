import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tkinter import *
from tkinter import ttk, messagebox

# -----------------------------
# Load and prepare the dataset
# -----------------------------
data = pd.read_csv("sales_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Revenue'] = data['Unit_Price'] * data['Quantity']

# Train the model
X = data[['Unit_Price', 'Quantity']]
y = data['Revenue']
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# GUI Functions
# -----------------------------

def predict_revenue():
    try:
        price = float(entry_price.get())
        qty = int(entry_qty.get())
        input_df = pd.DataFrame([[price, qty]], columns=['Unit_Price', 'Quantity'])
        pred = model.predict(input_df)
        result_label.config(text=f"📈 Predicted Revenue: ${pred[0]:.2f}", fg="green")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

def show_chart():
    product_revenue = data.groupby('Product')['Revenue'].sum()
    product_revenue.plot(kind='bar', title='💰 Revenue per Product', color='skyblue', edgecolor='black')
    plt.ylabel("Revenue ($)")
    plt.tight_layout()
    plt.show()

# -----------------------------
# GUI Setup (Tkinter)
# -----------------------------

root = Tk()
root.title("💼 Smart Sales Analyzer")
root.geometry("450x500")
root.configure(bg="#f4f6f7")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
style.configure("TLabel", font=("Segoe UI", 10), background="#f4f6f7")

# Title
Label(root, text="🧠 Smart Sales Analyzer", font=("Segoe UI", 14, "bold"), fg="#2c3e50", bg="#f4f6f7").pack(pady=10)

# Input Frame
frame = Frame(root, bg="#f4f6f7")
frame.pack(pady=10)

Label(frame, text="Unit Price ($):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
entry_price = Entry(frame)
entry_price.grid(row=0, column=1, padx=5, pady=5)

Label(frame, text="Quantity:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
entry_qty = Entry(frame)
entry_qty.grid(row=1, column=1, padx=5, pady=5)

# Buttons
Button(root, text="🔮 Predict Revenue", bg="#2980b9", fg="white", font=("Segoe UI", 10, "bold"),
       command=predict_revenue).pack(pady=5)

Button(root, text="📊 Show Revenue Chart", bg="#27ae60", fg="white", font=("Segoe UI", 10, "bold"),
       command=show_chart).pack(pady=5)

# -----------------------------
# Product Price Table
# -----------------------------
Label(root, text="🧾 Product Price List", font=("Segoe UI", 12, "bold"), bg="#f4f6f7", fg="#2c3e50").pack(pady=(10, 0))

tree_frame = Frame(root, bg="#f4f6f7")
tree_frame.pack()

columns = ('Product', 'Unit Price')
tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=6)
tree.heading('Product', text='Product')
tree.heading('Unit Price', text='Unit Price ($)')

# Sample product prices
product_prices = {
    'Laptop': 700,
    'Mouse': 25,
    'Keyboard': 35,
    'Monitor': 150,
    'Printer': 120,
    'Headphones': 50
}

# Insert rows into Treeview
for product, price in product_prices.items():
    tree.insert('', END, values=(product, price))

tree.pack(padx=10, pady=5)

# -----------------------------
# Result Display
# -----------------------------
result_label = Label(root, text="", font=("Segoe UI", 12, "bold"), bg="#f4f6f7")
result_label.pack(pady=10)

root.mainloop()
