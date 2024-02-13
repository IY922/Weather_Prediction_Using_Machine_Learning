import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from threading import Thread
from tkcalendar import DateEntry
from prophet import Prophet


class WeatherPredictionApp:
    def __init__(self, root_window):
        self.current_graph_label = None
        self.percentage_label = None
        self.progressbar = None
        self.entry_num_days = None
        self.date_picker_start_date = None
        self.entry_data_file = None
        self.root = root_window
        self.root.title("Weather Prediction App By YASIR")
        self.loading = tk.BooleanVar()
        self.loading.set(False)
        self.total_graphs = 15
        self.progress_value = 0
        self.create_widgets()

    def create_widgets(self):
        title_label = ttk.Label(self.root, text="Weather Prediction", font=('Helvetica', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, pady=15)

        ttk.Label(self.root, text="Select Data File:", font=('Helvetica', 12)).grid(row=1, column=0, padx=5, pady=10)
        self.entry_data_file = ttk.Entry(self.root, state="readonly", width=40, font=('Helvetica', 12))
        self.entry_data_file.grid(row=1, column=1, padx=5, pady=10)

        ttk.Button(self.root, text="Browse", command=self.browse_data_file).grid(row=1, column=2, padx=5, pady=10)

        ttk.Label(self.root, text="Select Start Date:", font=('Helvetica', 12)).grid(row=2, column=0, padx=5, pady=10)

        self.date_picker_start_date = DateEntry(self.root, width=17, background='Blue', foreground='white',
                                                date_pattern='dd-mm-yyyy', font=('Helvetica', 12))
        self.date_picker_start_date.grid(row=2, column=1, padx=5, pady=10)

        ttk.Label(self.root, text="Number of Days:", font=('Helvetica', 12)).grid(row=3, column=0, padx=5, pady=10)

        self.entry_num_days = ttk.Entry(self.root, width=10, font=('Helvetica', 12))
        self.entry_num_days.grid(row=3, column=1, padx=5, pady=10)

        style = ttk.Style()
        style.configure("TButton", font=('Helvetica', 14))

        ttk.Button(self.root, text="Start Prediction", command=self.start_prediction,
                   style="TButton").grid(row=4, column=0, columnspan=4, pady=15)

        self.progressbar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progressbar.grid(row=5, column=1, columnspan=4, pady=15)
        self.progressbar.grid_remove()

        self.percentage_label = ttk.Label(self.root, text="", font=('Helvetica', 12))
        self.percentage_label.grid(row=5, column=2, pady=15)
        self.percentage_label.grid_remove()

        self.current_graph_label = ttk.Label(self.root, text="", font=('Helvetica', 12))
        self.current_graph_label.grid(row=5, column=0, pady=15)
        self.current_graph_label.grid_remove()

    def update_progressbar(self, value):
        if self.loading.get():
            progress_value = int((value / self.total_graphs) * 100)
            self.progressbar.configure(mode="determinate", maximum=100, value=progress_value)
            self.percentage_label.config(text=f"{progress_value}%")
            self.percentage_label.grid()
            self.current_graph_label.config(text=f"Graph {value}/{self.total_graphs}")
            self.current_graph_label.grid()
            self.progress_value = progress_value

            if value < self.total_graphs:
                self.root.after(100, self.update_progressbar, value + 1)

            else:
                self.progressbar.configure(value=self.progress_value, mode="determinate")
                self.percentage_label.grid_remove()
                self.current_graph_label.grid_remove()

    def browse_data_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.entry_data_file.configure(state="normal")
        self.entry_data_file.delete(0, tk.END)
        self.entry_data_file.insert(0, file_path)
        self.entry_data_file.configure(state="readonly")

        if file_path:
            try:
                df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)

                if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                month_year_data = df['Date'].dt.to_period('M').iloc[0]
                self.date_picker_start_date.allowed_periods = [month_year_data]
            except Exception as e:
                # Handle the exception (e.g., show an error message)
                print(f"Error: {e}")

    def start_prediction(self):
        self.loading.set(True)
        self.progressbar.grid()

        data_file = self.entry_data_file.get()
        start_date = pd.to_datetime(self.date_picker_start_date.get_date()).strftime('%Y-%m-%d')
        num_days = int(self.entry_num_days.get())
        prediction_thread = Thread(target=self.run_prediction, args=(data_file, start_date, num_days))
        prediction_thread.start()

        self.update_progressbar(1)

    def run_prediction(self, data_file, start_date, num_days):
        df = pd.read_csv(data_file, parse_dates=['Date'], dayfirst=True)
        unique_columns = df.columns.difference(['Date']).tolist()
        rows = 5
        cols = 3
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=unique_columns, shared_xaxes=True,
                            vertical_spacing=0.1, horizontal_spacing=0.1)

        for i, column in enumerate(unique_columns):
            row = i // cols + 1
            col = i % cols + 1
            df_filtered = df[df['Date'] >= start_date]
            model = Prophet()
            df_filtered = df_filtered.rename(columns={column: 'y', 'Date': 'ds'})
            model.fit(df_filtered)
            future = model.make_future_dataframe(periods=num_days, include_history=False)
            forecast = model.predict(future)
            trace = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prediction')
            fig.add_trace(trace, row=row, col=col)
            trace_actual = go.Scatter(x=df_filtered['ds'], y=df_filtered['y'], mode='markers', name='Actual')
            fig.add_trace(trace_actual, row=row, col=col)

            self.update_progressbar(i + 2)

        fig.update_layout(height=400 * rows, showlegend=True)
        self.loading.set(False)
        self.progressbar.grid_remove()
        fig.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherPredictionApp(root)
    root.mainloop()
