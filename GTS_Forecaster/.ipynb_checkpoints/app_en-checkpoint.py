import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import subprocess
import threading
import locale

class AdvancedParamsDialog(simpledialog.Dialog):
    def __init__(self, parent, model_type):
        self.model_type = model_type
        super().__init__(parent, f"{model_type} Advanced Parameters")

    def body(self, master):
        self.params = {}
        row = 0
        
        # Common parameters
        ttk.Label(master, text="Dropout Rate:").grid(row=row, sticky=tk.W)
        self.dropout = ttk.Entry(master)
        self.dropout.insert(0, "0.2")
        self.dropout.grid(row=row, column=1)
        self.params["--dropout"] = self.dropout
        row += 1

        ttk.Label(master, text="Hidden Layer Dimension:").grid(row=row, sticky=tk.W)
        self.hidden_dim = ttk.Spinbox(master, from_=8, to=256, increment=8)
        self.hidden_dim.set(32)
        self.hidden_dim.grid(row=row, column=1)
        self.params["--hidden_dim"] = self.hidden_dim
        row += 1

        ttk.Label(master, text="Number of Layers:").grid(row=row, sticky=tk.W)
        self.n_layers = ttk.Spinbox(master, from_=1, to=10)
        self.n_layers.set(2)
        self.n_layers.grid(row=row, column=1)
        self.params["--n_layers"] = self.n_layers
        row += 1

        # Model-specific parameters
        if "TCN" in self.model_type:
            ttk.Label(master, text="Kernel Size:").grid(row=row, sticky=tk.W)
            self.kernel_size = ttk.Spinbox(master, from_=3, to=9, increment=2)
            self.kernel_size.set(3)
            self.kernel_size.grid(row=row, column=1)
            self.params["--kernel_size"] = self.kernel_size
            row += 1

            ttk.Label(master, text="Number of Channels:").grid(row=row, sticky=tk.W)
            self.num_channels = ttk.Entry(master)
            self.num_channels.insert(0, "25,50,25")
            self.num_channels.grid(row=row, column=1)
            self.params["--num_channels"] = self.num_channels
            row += 1

        if "Transformer" in self.model_type:
            ttk.Label(master, text="Number of Attention Heads:").grid(row=row, sticky=tk.W)
            self.num_heads = ttk.Spinbox(master, from_=1, to=8, increment=1)
            self.num_heads.set(4)
            self.num_heads.grid(row=row, column=1)
            self.params["--num_heads"] = self.num_heads
            row += 1

            ttk.Label(master, text="Feedforward Dimension:").grid(row=row, sticky=tk.W)
            self.ff_dim = ttk.Spinbox(master, from_=64, to=512, increment=64)
            self.ff_dim.set(32)
            self.ff_dim.grid(row=row, column=1)
            self.params["--hidden_space"] = self.ff_dim
            row += 1

        return master

    def apply(self):
        self.result = {key: widget.get() for key, widget in self.params.items()}

class TSForecastGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Geodetic Time Series Forecasting Toolbox in Python/GTS_Forecaster V1.0")
        self.geometry("1200x900")
        self.file_path = tk.StringVar()
        self.advanced_args = {}
        self.encoding = locale.getpreferredencoding()
        self.create_widgets()

    def create_widgets(self):
        # File selection area
        file_frame = ttk.LabelFrame(self, text="Data File Configuration")
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="CSV File Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_path, width=80).grid(row=0, column=1)
        ttk.Button(file_frame, text="Browse", command=self.load_file).grid(row=0, column=2)

        # Feature selection area
        feature_frame = ttk.LabelFrame(self, text="Feature Selection (Multiple Selection Allowed)")
        feature_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

        # Available features list
        ttk.Label(feature_frame, text="Available Features").grid(row=0, column=0)
        self.available_list = tk.Listbox(feature_frame, selectmode=tk.EXTENDED, height=15,
                                       exportselection=0, bg="#F0F0F0")
        self.available_list.grid(row=1, column=0, sticky=tk.NSEW, padx=5)

        # Operation buttons
        btn_frame = ttk.Frame(feature_frame)
        btn_frame.grid(row=1, column=1, padx=5)
        ttk.Button(btn_frame, text="Add to Input →", command=lambda: self.add_features("input")).pack(pady=2)
        ttk.Button(btn_frame, text="Remove from Input ←", command=lambda: self.remove_features("input")).pack(pady=2)
        ttk.Button(btn_frame, text="Add to Output →", command=lambda: self.add_features("output")).pack(pady=2)
        ttk.Button(btn_frame, text="Remove from Output ←", command=lambda: self.remove_features("output")).pack(pady=2)

        # Input features list
        ttk.Label(feature_frame, text="Input Features").grid(row=0, column=2)
        self.input_list = tk.Listbox(feature_frame, selectmode=tk.EXTENDED, height=15, bg="#E6F3FF")
        self.input_list.grid(row=1, column=2, sticky=tk.NSEW, padx=5)

        # Output features list
        ttk.Label(feature_frame, text="Output Features").grid(row=0, column=3)
        self.output_list = tk.Listbox(feature_frame, selectmode=tk.EXTENDED, height=15, bg="#FFE6E6")
        self.output_list.grid(row=1, column=3, sticky=tk.NSEW, padx=5)

        # Grid configuration
        for i in range(4):
            feature_frame.columnconfigure(i, weight=1)
        feature_frame.rowconfigure(1, weight=1)

        # Core parameters area
        param_frame = ttk.LabelFrame(self, text="Core Parameter Configuration")
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # First row parameters
        ttk.Label(param_frame, text="Model Type:").grid(row=0, column=0, sticky=tk.W)
        self.model_type = ttk.Combobox(param_frame, values=[
            "LSTM", "LSTM_ekan", "TCN", "TCN_ekan", 
            "Transformer", "Transformer_ekan", "BiLSTM", 
            "BiLSTM_ekan", "GRU", "GRU_ekan", "LSTM_Attention",
            "VanillaLSTM", "ConvLSTM", "CustomLSTM"
        ], state="readonly", width=15)
        self.model_type.current(0)
        self.model_type.grid(row=0, column=1)
        
        ttk.Label(param_frame, text="Window Size:").grid(row=0, column=2, sticky=tk.W)
        self.window_size = ttk.Spinbox(param_frame, from_=1, to=365, width=8)
        self.window_size.set(20)
        self.window_size.grid(row=0, column=3)
        
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=4, sticky=tk.W)
        self.batch_size = ttk.Spinbox(param_frame, from_=8, to=256, width=8)
        self.batch_size.set(32)
        self.batch_size.grid(row=0, column=5)
        
        # New training epochs control
        ttk.Label(param_frame, text="Training Epochs:").grid(row=0, column=6, sticky=tk.W)
        self.epochs = ttk.Spinbox(param_frame, from_=1, to=1000, width=8)
        self.epochs.set(100)
        self.epochs.grid(row=0, column=7)


        # Second row parameters (continued)
        ttk.Label(param_frame, text="Learning Rate:").grid(row=1, column=2, sticky=tk.W)
        self.lr = ttk.Entry(param_frame, width=10)
        self.lr.insert(0, "0.001")  # Set default learning rate
        self.lr.grid(row=1, column=3)

        ttk.Label(param_frame, text="Train/Test Split Ratio:").grid(row=1, column=4, sticky=tk.W)
        self.train_test_ratio = ttk.Spinbox(param_frame, from_=0.1, to=0.5, increment=0.05, width=6)
        self.train_test_ratio.set(0.2)
        self.train_test_ratio.grid(row=1, column=5)

        # Third row parameters
        ttk.Label(param_frame, text="Prediction Start:").grid(row=2, column=0, sticky=tk.W)
        self.predict_start = ttk.Spinbox(param_frame, from_=0, to=100000, width=8)
        self.predict_start.set(20)
        self.predict_start.grid(row=2, column=1)

        ttk.Label(param_frame, text="Prediction End:").grid(row=2, column=2, sticky=tk.W)
        self.predict_end = ttk.Spinbox(param_frame, from_=0, to=100000, width=8)
        self.predict_end.set(100)
        self.predict_end.grid(row=2, column=3)

        # Fourth row parameters
        ttk.Label(param_frame, text="Prediction Length:").grid(row=3, column=0, sticky=tk.W)
        self.predict_nums = ttk.Entry(param_frame, width=15)
        self.predict_nums.insert(0, "30,60,360")  # Set default prediction lengths
        self.predict_nums.grid(row=3, column=1)

        self.use_early_stop = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Enable Early Stopping", variable=self.use_early_stop).grid(row=3, column=2)
        
        ttk.Label(param_frame, text="Early Stopping Patience:").grid(row=3, column=3, sticky=tk.W)
        self.patience = ttk.Spinbox(param_frame, from_=10, to=500, width=8)
        self.patience.set(250)
        self.patience.grid(row=3, column=4)
        
        ttk.Label(param_frame, text="Minimum Change:").grid(row=3, column=5, sticky=tk.W)
        self.delta = ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.01, width=6)
        self.delta.set(0.0)
        self.delta.grid(row=3, column=6)

        # Advanced parameters button
        ttk.Button(param_frame, text="Advanced Parameters...", command=self.show_advanced_params).grid(row=4, column=6)

        # Console area
        console_frame = ttk.LabelFrame(self, text="Console Output")
        console_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        self.console = tk.Text(console_frame, wrap=tk.WORD, height=15, bg="#1E1E1E", fg="white")
        scroll = ttk.Scrollbar(console_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Run button
        run_btn = ttk.Button(console_frame, text="Start Training", command=self.run_training)
        run_btn.pack(side=tk.BOTTOM, pady=5)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.file_path.set(path)
            try:
                df = pd.read_csv(path, nrows=1)
                self.available_list.delete(0, tk.END)
                for col in df.columns:
                    self.available_list.insert(tk.END, col)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {str(e)}")

    def add_features(self, target):
        selected = self.available_list.curselection()
        target_list = self.input_list if target == "input" else self.output_list
        for idx in selected:
            item = self.available_list.get(idx)
            target_list.insert(tk.END, item)

    def remove_features(self, target):
        target_list = self.input_list if target == "input" else self.output_list
        selected = target_list.curselection()
        for idx in reversed(selected):
            target_list.delete(idx)

    def show_advanced_params(self):
        model = self.model_type.get()
        dialog = AdvancedParamsDialog(self, model)
        if hasattr(dialog, 'result'):
            self.advanced_args = dialog.result

    def validate_params(self):
        try:
            if not self.file_path.get():
                raise ValueError("Please select a data file")
            if self.input_list.size() == 0:
                raise ValueError("At least one input feature must be selected")
            if self.output_list.size() == 0:
                raise ValueError("At least one output feature must be selected")
            
            start = int(self.predict_start.get())
            end = int(self.predict_end.get())
            if start >= end:
                raise ValueError("Prediction start index must be less than end index")

            # Validate numerical parameters
            float(self.lr.get())
            [int(x) for x in self.predict_nums.get().split(',')]
            float(self.train_test_ratio.get())

            # Validate model-specific parameters
            if "TCN" in self.model_type.get():
                if '--num_channels' in self.advanced_args:
                    channels = list(map(int, self.advanced_args['--num_channels'].split(',')))
                    if len(channels) < 2:
                        raise ValueError("TCN requires at least 2 channel layers")
            return True
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Parameter validation failed: {str(e)}")
            return False

    def run_training(self):
        if not self.validate_params():
            return

        cmd = [
            "python", "71.py",
            "--path", self.file_path.get(),
            "--input_features", ",".join(self.input_list.get(0, tk.END)),
            "--output_features", ",".join(self.output_list.get(0, tk.END)),
            "--model_name", self.model_type.get(),
            "--window_size", self.window_size.get(),
            "--batch_size", self.batch_size.get(),
            "--train_test_ratio", self.train_test_ratio.get(),
            "--predict_start", self.predict_start.get(),
            "--predict_end", self.predict_end.get(),
            "--num_epochs", self.epochs.get(),
            "--lr", self.lr.get(),
            "--predict_nums", " ".join(self.predict_nums.get().split(','))
        ]

        # Early stopping parameters
        if self.use_early_stop.get():
            cmd += [
                "--use_early_stopping",
                "--patience", self.patience.get(),
                "--delta", self.delta.get()
            ]

        # Advanced parameters
        for key, value in self.advanced_args.items():
            if key == "--num_channels":
                cmd += [key, " ".join(value.split(','))]
            else:
                cmd += [key, str(value)]

        # Console output configuration
        self.console.configure(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.insert(tk.END, "$ " + " ".join(cmd) + "\n\n")
        self.console.configure(state=tk.DISABLED)

        def run_in_thread():
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    encoding=locale.getpreferredencoding(),
                    errors='replace'
                )
                
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        self.after(0, self.append_output, line)
                
                exit_code = process.poll()
                if exit_code == 0:
                    self.append_output("\nTraining completed!")
                else:
                    self.append_output(f"\nProcess exited abnormally, code {exit_code}")
            except Exception as e:
                self.append_output(f"\nError: {str(e)}")

        threading.Thread(target=run_in_thread, daemon=True).start()

    def append_output(self, text):
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)
        self.update_idletasks()

if __name__ == "__main__":
    app = TSForecastGUI()
    app.mainloop()