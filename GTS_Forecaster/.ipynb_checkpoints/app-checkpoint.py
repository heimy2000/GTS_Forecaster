import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import subprocess
import threading
import locale

class AdvancedParamsDialog(simpledialog.Dialog):
    def __init__(self, parent, model_type):
        self.model_type = model_type
        super().__init__(parent, f"{model_type} 高级参数设置")

    def body(self, master):
        self.params = {}
        row = 0
        
        # 公共参数
        ttk.Label(master, text="Dropout率:").grid(row=row, sticky=tk.W)
        self.dropout = ttk.Entry(master)
        self.dropout.insert(0, "0.2")
        self.dropout.grid(row=row, column=1)
        self.params["--dropout"] = self.dropout
        row += 1

        ttk.Label(master, text="隐藏层维度:").grid(row=row, sticky=tk.W)
        self.hidden_dim = ttk.Spinbox(master, from_=8, to=256, increment=8)
        self.hidden_dim.set(32)
        self.hidden_dim.grid(row=row, column=1)
        self.params["--hidden_dim"] = self.hidden_dim
        row += 1

        ttk.Label(master, text="网络层数:").grid(row=row, sticky=tk.W)
        self.n_layers = ttk.Spinbox(master, from_=1, to=10)
        self.n_layers.set(2)
        self.n_layers.grid(row=row, column=1)
        self.params["--n_layers"] = self.n_layers
        row += 1

        # 模型特定参数
        if "TCN" in self.model_type:
            ttk.Label(master, text="卷积核大小:").grid(row=row, sticky=tk.W)
            self.kernel_size = ttk.Spinbox(master, from_=3, to=9, increment=2)
            self.kernel_size.set(3)
            self.kernel_size.grid(row=row, column=1)
            self.params["--kernel_size"] = self.kernel_size
            row += 1

            ttk.Label(master, text="通道数:").grid(row=row, sticky=tk.W)
            self.num_channels = ttk.Entry(master)
            self.num_channels.insert(0, "25,50,25")
            self.num_channels.grid(row=row, column=1)
            self.params["--num_channels"] = self.num_channels
            row += 1

        if "Transformer" in self.model_type:
            ttk.Label(master, text="注意力头数:").grid(row=row, sticky=tk.W)
            self.num_heads = ttk.Spinbox(master, from_=1, to=8, increment=1)
            self.num_heads.set(4)
            self.num_heads.grid(row=row, column=1)
            self.params["--num_heads"] = self.num_heads
            row += 1

            ttk.Label(master, text="前馈维度:").grid(row=row, sticky=tk.W)
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
        # 文件选择区域
        file_frame = ttk.LabelFrame(self, text="数据文件配置")
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="CSV文件路径:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_path, width=80).grid(row=0, column=1)
        ttk.Button(file_frame, text="浏览", command=self.load_file).grid(row=0, column=2)

        # 特征选择区域
        feature_frame = ttk.LabelFrame(self, text="特征选择（可重复选择）")
        feature_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

        # 可用特征列表
        ttk.Label(feature_frame, text="可用特征").grid(row=0, column=0)
        self.available_list = tk.Listbox(feature_frame, selectmode=tk.EXTENDED, height=15,
                                       exportselection=0, bg="#F0F0F0")
        self.available_list.grid(row=1, column=0, sticky=tk.NSEW, padx=5)

        # 操作按钮
        btn_frame = ttk.Frame(feature_frame)
        btn_frame.grid(row=1, column=1, padx=5)
        ttk.Button(btn_frame, text="添加输入 →", command=lambda: self.add_features("input")).pack(pady=2)
        ttk.Button(btn_frame, text="移除输入 ←", command=lambda: self.remove_features("input")).pack(pady=2)
        ttk.Button(btn_frame, text="添加输出 →", command=lambda: self.add_features("output")).pack(pady=2)
        ttk.Button(btn_frame, text="移除输出 ←", command=lambda: self.remove_features("output")).pack(pady=2)

        # 输入特征列表
        ttk.Label(feature_frame, text="输入特征").grid(row=0, column=2)
        self.input_list = tk.Listbox(feature_frame, selectmode=tk.EXTENDED, height=15, bg="#E6F3FF")
        self.input_list.grid(row=1, column=2, sticky=tk.NSEW, padx=5)

        # 输出特征列表
        ttk.Label(feature_frame, text="输出特征").grid(row=0, column=3)
        self.output_list = tk.Listbox(feature_frame, selectmode=tk.EXTENDED, height=15, bg="#FFE6E6")
        self.output_list.grid(row=1, column=3, sticky=tk.NSEW, padx=5)

        # 网格配置
        for i in range(4):
            feature_frame.columnconfigure(i, weight=1)
        feature_frame.rowconfigure(1, weight=1)

        # 核心参数区域
        param_frame = ttk.LabelFrame(self, text="核心参数配置")
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 第一行参数
        ttk.Label(param_frame, text="模型类型:").grid(row=0, column=0, sticky=tk.W)
        self.model_type = ttk.Combobox(param_frame, values=[
            "LSTM", "LSTM_ekan", "TCN", "TCN_ekan", 
            "Transformer", "Transformer_ekan", "BiLSTM", 
            "BiLSTM_ekan", "GRU", "GRU_ekan", "LSTM_Attention",
            "VanillaLSTM", "ConvLSTM", "CustomLSTM"
        ], state="readonly", width=15)
        self.model_type.current(0)
        self.model_type.grid(row=0, column=1)
        
        ttk.Label(param_frame, text="窗口大小:").grid(row=0, column=2, sticky=tk.W)
        self.window_size = ttk.Spinbox(param_frame, from_=1, to=365, width=8)
        self.window_size.set(20)
        self.window_size.grid(row=0, column=3)
        
        ttk.Label(param_frame, text="批大小:").grid(row=0, column=4, sticky=tk.W)
        self.batch_size = ttk.Spinbox(param_frame, from_=8, to=256, width=8)
        self.batch_size.set(32)
        self.batch_size.grid(row=0, column=5)
        
        # 新增训练轮数控件
        ttk.Label(param_frame, text="训练轮数:").grid(row=0, column=6, sticky=tk.W)
        self.epochs = ttk.Spinbox(param_frame, from_=1, to=1000, width=8)
        self.epochs.set(100)
        self.epochs.grid(row=0, column=7)


        # 第二行参数（续）
        ttk.Label(param_frame, text="学习率:").grid(row=1, column=2, sticky=tk.W)
        self.lr = ttk.Entry(param_frame, width=10)
        self.lr.insert(0, "0.001")  # 设置默认学习率
        self.lr.grid(row=1, column=3)

        ttk.Label(param_frame, text="分割比例:").grid(row=1, column=4, sticky=tk.W)
        self.train_test_ratio = ttk.Spinbox(param_frame, from_=0.1, to=0.5, increment=0.05, width=6)
        self.train_test_ratio.set(0.2)
        self.train_test_ratio.grid(row=1, column=5)

        # 第三行参数
        ttk.Label(param_frame, text="预测起始:").grid(row=2, column=0, sticky=tk.W)
        self.predict_start = ttk.Spinbox(param_frame, from_=0, to=100000, width=8)
        self.predict_start.set(20)
        self.predict_start.grid(row=2, column=1)

        ttk.Label(param_frame, text="预测结束:").grid(row=2, column=2, sticky=tk.W)
        self.predict_end = ttk.Spinbox(param_frame, from_=0, to=100000, width=8)
        self.predict_end.set(100)
        self.predict_end.grid(row=2, column=3)

        # 第四行参数
        ttk.Label(param_frame, text="预测长度:").grid(row=3, column=0, sticky=tk.W)
        self.predict_nums = ttk.Entry(param_frame, width=15)
        self.predict_nums.insert(0, "30,60,360")  # 设置默认预测长度
        self.predict_nums.grid(row=3, column=1)

        self.use_early_stop = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="启用早停", variable=self.use_early_stop).grid(row=3, column=2)
        
        ttk.Label(param_frame, text="早停耐心:").grid(row=3, column=3, sticky=tk.W)
        self.patience = ttk.Spinbox(param_frame, from_=10, to=500, width=8)
        self.patience.set(250)
        self.patience.grid(row=3, column=4)
        
        ttk.Label(param_frame, text="最小变化:").grid(row=3, column=5, sticky=tk.W)
        self.delta = ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.01, width=6)
        self.delta.set(0.0)
        self.delta.grid(row=3, column=6)

        # 高级参数按钮
        ttk.Button(param_frame, text="高级参数...", command=self.show_advanced_params).grid(row=4, column=6)

        # 控制台区域
        console_frame = ttk.LabelFrame(self, text="运行控制台")
        console_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        self.console = tk.Text(console_frame, wrap=tk.WORD, height=15, bg="#1E1E1E", fg="white")
        scroll = ttk.Scrollbar(console_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 运行按钮
        run_btn = ttk.Button(console_frame, text="开始训练", command=self.run_training)
        run_btn.pack(side=tk.BOTTOM, pady=5)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv")])
        if path:
            self.file_path.set(path)
            try:
                df = pd.read_csv(path, nrows=1)
                self.available_list.delete(0, tk.END)
                for col in df.columns:
                    self.available_list.insert(tk.END, col)
            except Exception as e:
                messagebox.showerror("错误", f"文件读取失败: {str(e)}")

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
                raise ValueError("请选择数据文件")
            if self.input_list.size() == 0:
                raise ValueError("必须选择至少一个输入特征")
            if self.output_list.size() == 0:
                raise ValueError("必须选择至少一个输出特征")
            
            start = int(self.predict_start.get())
            end = int(self.predict_end.get())
            if start >= end:
                raise ValueError("预测起始索引必须小于结束索引")

            # 验证数值参数
            float(self.lr.get())
            [int(x) for x in self.predict_nums.get().split(',')]
            float(self.train_test_ratio.get())

            # 验证模型特定参数
            if "TCN" in self.model_type.get():
                if '--num_channels' in self.advanced_args:
                    channels = list(map(int, self.advanced_args['--num_channels'].split(',')))
                    if len(channels) < 2:
                        raise ValueError("TCN通道数至少需要2个层级")
            return True
        except ValueError as e:
            messagebox.showerror("参数错误", f"参数验证失败: {str(e)}")
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

        # 早停参数
        if self.use_early_stop.get():
            cmd += [
                "--use_early_stopping",
                "--patience", self.patience.get(),
                "--delta", self.delta.get()
            ]

        # 高级参数
        for key, value in self.advanced_args.items():
            if key == "--num_channels":
                cmd += [key, " ".join(value.split(','))]
            else:
                cmd += [key, str(value)]

        # 控制台输出配置
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
                    self.append_output("\n训练完成！")
                else:
                    self.append_output(f"\n进程异常退出，代码 {exit_code}")
            except Exception as e:
                self.append_output(f"\n错误: {str(e)}")

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
