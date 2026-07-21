import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from main import run_pipeline

class TopicModelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NMF / SVD / LDA 文本主题分析系统")
        self.root.geometry("1300x800")
        
        print("正在后台加载数据并训练模型，请稍候...")
        self.raw_texts, self.targets, self.target_names, self.results = run_pipeline(n_topics=4, n_features=1000)
        self.model_names = list(self.results.keys())
        print("模型加载完毕！正在启动图形界面...")

        self.current_model = tk.StringVar(value=self.model_names[0])
        self.current_doc_id = tk.IntVar(value=0)
        self.max_doc_id = len(self.raw_texts) - 1

        self._build_ui()
        
        self.update_view()

    def _build_ui(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="选择算法模型:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        model_combo = ttk.Combobox(control_frame, textvariable=self.current_model, values=self.model_names, state="readonly", width=20)
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind("<<ComboboxSelected>>", lambda e: self.update_view())

        ttk.Label(control_frame, text=f"   输入或拖动选择文章 (ID: 0 - {self.max_doc_id}):", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        doc_spinbox = ttk.Spinbox(control_frame, from_=0, to=self.max_doc_id, textvariable=self.current_doc_id, width=8, command=self.update_view)
        doc_spinbox.pack(side=tk.LEFT, padx=5)
        doc_spinbox.bind('<Return>', lambda e: self.update_view())
        
        doc_slider = tk.Scale(control_frame, from_=0, to=self.max_doc_id, variable=self.current_doc_id, orient=tk.HORIZONTAL, length=250, command=lambda v: self.update_view())
        doc_slider.pack(side=tk.LEFT, padx=5)

        left_frame = ttk.Frame(self.root, padding=10, width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        
        ttk.Label(left_frame, text="📄 原始文章内容", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.text_box = tk.Text(left_frame, wrap=tk.WORD, width=60, height=18, font=("Consolas", 10))
        self.text_box.pack(pady=5, fill=tk.BOTH, expand=True)

        left_title_frame = ttk.Frame(left_frame)
        left_title_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.truth_label_var = tk.StringVar()
        ttk.Label(left_title_frame, textvariable=self.truth_label_var, font=("Arial", 12, "bold"), foreground="blue").pack(side=tk.LEFT)
        
        btn_export_doc = ttk.Button(left_title_frame, text="💾 导出左图", command=self.export_doc_plot)
        btn_export_doc.pack(side=tk.RIGHT)
        
        self.fig_doc, self.ax_doc = plt.subplots(figsize=(5, 3))
        self.canvas_doc = FigureCanvasTkAgg(self.fig_doc, master=left_frame)
        self.canvas_doc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        right_title_frame = ttk.Frame(right_frame)
        right_title_frame.pack(fill=tk.X)
        ttk.Label(right_title_frame, text="🗂️ 全局四大主题的核心词汇 (Top 10)", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        btn_export_topics = ttk.Button(right_title_frame, text="💾 导出右图", command=self.export_topics_plot)
        btn_export_topics.pack(side=tk.RIGHT)
        
        self.fig_topics, self.axes_topics = plt.subplots(2, 2, figsize=(8, 6))
        self.fig_topics.tight_layout(pad=3.0)
        self.canvas_topics = FigureCanvasTkAgg(self.fig_topics, master=right_frame)
        self.canvas_topics.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    def update_view(self, *args):
        try:
            doc_id = self.current_doc_id.get()
        except tk.TclError:
            return
            
        if doc_id < 0 or doc_id > self.max_doc_id:
            return

        model = self.current_model.get()

        self.text_box.config(state=tk.NORMAL)
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, self.raw_texts[doc_id])
        self.text_box.config(state=tk.DISABLED)
        
        true_label_id = self.targets[doc_id]
        true_label_name = self.target_names[true_label_id]
        self.truth_label_var.set(f"📊 主题归一化构成比例  [真实分类: {true_label_name}]")

        raw_doc_dist = self.results[model]['doc_topic'][doc_id]
        
        dist_sum = np.sum(np.abs(raw_doc_dist))
        if dist_sum > 0:
            doc_dist = raw_doc_dist / dist_sum
        else:
            doc_dist = raw_doc_dist

        self.ax_doc.clear()
        topic_labels = [f"Topic {i}" for i in range(len(doc_dist))]
        colors_doc = ['#ff9999' if val < 0 else '#66b3ff' for val in doc_dist]
        
        self.ax_doc.bar(topic_labels, doc_dist, color=colors_doc, edgecolor='black', alpha=0.8)
        self.ax_doc.set_ylabel("归一化比例 (Proportion)")
        self.ax_doc.grid(axis='y', linestyle='--', alpha=0.7)
        
        y_max = max(1.0, np.max(doc_dist) + 0.1)
        y_min = min(0.0, np.min(doc_dist) - 0.1)
        self.ax_doc.set_ylim(y_min, y_max)

        if "SVD" in model:
            self.ax_doc.axhline(0, color='black', linewidth=1)
        
        self.fig_doc.tight_layout()
        self.canvas_doc.draw()

        topic_words = self.results[model]['topic_word']
        feature_names = self.results[model]['features']
        
        for ax in self.axes_topics.flatten():
            ax.clear()

        for topic_idx, topic_weights in enumerate(topic_words):
            ax = self.axes_topics.flatten()[topic_idx]
            
            top_indices = np.argsort(np.abs(topic_weights))[:-11:-1]
            top_features = [feature_names[i] for i in top_indices]
            top_values = topic_weights[top_indices]
            
            y_pos = np.arange(len(top_features))
            colors_topics = ['#ff6666' if val < 0 else '#99cc99' for val in top_values]
            
            ax.barh(y_pos, top_values, align='center', color=colors_topics, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features, fontsize=9)
            ax.invert_yaxis()
            ax.set_title(f"Topic {topic_idx}", fontsize=11)
            if "SVD" in model:
                ax.axvline(0, color='black', linewidth=1)

        self.fig_topics.tight_layout()
        self.canvas_topics.draw()

    def export_doc_plot(self):
        model_name = self.current_model.get().split()[0]
        doc_id = self.current_doc_id.get()
        default_name = f"{model_name}_Doc{doc_id}_Normalized_Distribution.png"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=default_name,
            title="保存归一化文章主题分布图",
            filetypes=[("PNG 图像", "*.png"), ("PDF 文档", "*.pdf")]
        )
        if filepath:
            self.fig_doc.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("成功", f"图表已保存至:\n{filepath}")

    def export_topics_plot(self):
        model_name = self.current_model.get().split()[0]
        default_name = f"{model_name}_Global_Topics.png"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=default_name,
            title="保存全局主题图",
            filetypes=[("PNG 图像", "*.png"), ("PDF 文档", "*.pdf")]
        )
        if filepath:
            self.fig_topics.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("成功", f"图表已保存至:\n{filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = TopicModelingApp(root)
    root.mainloop()