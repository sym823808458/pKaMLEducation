import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageTk
from sklearn.tree import DecisionTreeRegressor, _tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# --- Stepwise knowledge content (English, simple) ---
KNOWLEDGE_STEPS = [
    # 1. Acid-base and pKa basics
    """**Acid Dissociation Equilibrium and pKa**
The acidity of a compound can be described by its ability to release a proton (H⁺).
For a generic acid (HA), the dissociation equilibrium is:
    HA ⇌ H⁺ + A⁻
The acid dissociation constant is:
    Kₐ = [H⁺][A⁻] / [HA]
pKa is defined as the negative logarithm:
    pKₐ = -log₁₀(Kₐ)
A lower pKa means stronger acid (more dissociation), a higher pKa means weaker acid.""",

    # 2. Measurement and experimental challenges
    """**How is pKa measured?**
For simple acids, pKa can be determined by titration, pH meters, or using acid-base indicators.
For complex organic acids, accurate measurement is often difficult due to low solubility, instability, or overlapping equilibria. In such cases, we often turn to computational or data-driven prediction methods like machine learning.""",

    # 3. Connection between structure and acidity
    """**What determines acidity and pKa?**
The acidity of a compound depends on its molecular structure:
- Electron-withdrawing groups increase acidity (lower pKa)
- Electron-donating groups decrease acidity (raise pKa)
- Aromatic rings, conjugation, and resonance can influence acidity
Chemists use these concepts to estimate or rationalize pKa, but quantitative prediction is challenging for diverse structures.""",

    # 4. Machine learning: definition and workflow
    """**What is Machine Learning?**
Machine learning is about building models that learn patterns from data, and can make predictions for new, unseen data.
General workflow:
1. Data collection & preprocessing
2. Define the task (e.g., regression or classification)
3. Choose a model (e.g., decision tree)
4. Train the model on existing data (training set)
5. Evaluate the model on new data (validation/test set)
6. Use the trained model to make predictions.""",

    # 5. Key terms in machine learning
    """**Key Terms in Machine Learning**
- Sample: One example (e.g., one molecule), described by features (inputs) and a label (output).
- Features: Quantitative properties (e.g., carbon count, aromatic rings, etc.).
- Label: The value to predict (e.g., pKa).
- Dataset: Many samples together.
- Training set: Used to teach the model.
- Validation/Test set: Used to check model performance on unseen data.""",

    # 6. Types of supervised learning
    """**Types of Supervised Learning**
- Regression: Predict a continuous value (e.g., pKa)
- Classification: Predict a category (e.g., acid or not)
In this tool, we focus on regression to predict pKa values.""",

    # 7. Model evaluation
    """**How do we evaluate models?**
Common metrics for regression:
- Mean Squared Error (MSE): Average squared difference between predicted and true values. Lower is better.
- Mean Absolute Error (MAE): Average absolute difference.
- R² Score: Measures how much variance is explained (1 = perfect, 0 = no better than mean).
We use cross-validation to get a reliable estimate of model performance.""",

    # 8. Decision tree regression: intuitive explanation
    """**What is a Decision Tree?**
A decision tree is like a flowchart: at each 'node', it asks a yes/no question about a feature (e.g., "Is carbon count > 5?").
Based on the answer, the sample goes left or right. At the end, the tree gives a prediction.
Decision trees are easy to interpret: you can see which features matter most, and why the prediction was made.
In this app, we use a regression tree with a maximum depth of 3 for clarity and interpretability.""",

    # 9. Ready to start!
    """You are now ready to start exploring pKa prediction with decision trees!
Please upload your dataset to begin."""
]

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive pKa Prediction Tool")
        self.dataset = None
        self.testset = None
        self.model = None
        self.features = []
        self.smiles_col = "SMILES"
        self.name_col = "Molecule Name"
        self.train_X = self.train_y = self.val_X = self.val_y = None
        self.val_idx = None
        self.knowledge_index = 0
        self.structure_window = None  # 分子结构弹窗
        self.selected_depth = 3  # 存储选择的深度
        self.create_knowledge_window()
        self.create_widgets()
 
    def create_knowledge_window(self):
        self.knowledge_win = tk.Toplevel(self.root)
        self.knowledge_win.title("Stepwise Knowledge Introduction")
        self.knowledge_win.geometry("600x400")
        
        # 创建滚动框架
        main_frame = tk.Frame(self.knowledge_win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.knowledge_text = tk.Text(scrollable_frame, width=70, height=12, wrap="word", font=("Arial", 11))
        self.knowledge_text.pack(padx=10, pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.next_btn = tk.Button(self.knowledge_win, text="Next", command=self.show_next_knowledge)
        self.next_btn.pack(pady=5)
        self.show_next_knowledge()
        
        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)

    def show_next_knowledge(self):
        self.knowledge_text.delete(1.0, tk.END)
        self.knowledge_text.insert(tk.END, KNOWLEDGE_STEPS[self.knowledge_index])
        self.knowledge_index += 1
        if self.knowledge_index >= len(KNOWLEDGE_STEPS):
            self.next_btn.config(text="Finish", command=self.close_knowledge)
    
    def close_knowledge(self):
        self.knowledge_win.destroy()
        self.upload_btn.config(state=tk.NORMAL)
        self.view_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
        self.tree_btn.config(state=tk.NORMAL)
        self.predict_btn.config(state=tk.NORMAL)

    def create_widgets(self):
        self.upload_btn = tk.Button(self.root, text="Upload CSV", command=self.upload_file, state=tk.DISABLED)
        self.upload_btn.pack(pady=8)
        self.view_btn = tk.Button(self.root, text="Dataset Visualization", command=self.visualize_dataset, state=tk.DISABLED)
        self.view_btn.pack(pady=8)
        self.train_btn = tk.Button(self.root, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_btn.pack(pady=8)
        self.tree_btn = tk.Button(self.root, text="Show Chemical Decision Tree", command=self.show_chem_tree, state=tk.DISABLED)
        self.tree_btn.pack(pady=8)
        self.predict_btn = tk.Button(self.root, text="Predict pKa", command=self.predict, state=tk.DISABLED)
        self.predict_btn.pack(pady=8)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            mask_missing = df.isnull().any(axis=1)
            if sum(mask_missing) > 0:
                self.testset = df.loc[mask_missing].reset_index(drop=True)
                self.dataset = df.loc[~mask_missing].reset_index(drop=True)
            else:
                self.dataset = df
                self.testset = None
            self.features = [col for col in df.columns if col not in [self.smiles_col, self.name_col, 'pKa']]
            self.view_btn.config(state=tk.NORMAL)
            self.train_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Info", "Dataset loaded successfully. Rows with missing features will be used as test set.")


    def visualize_dataset(self):
        win = tk.Toplevel(self.root)
        win.title("Dataset Visualization")
        columns = [self.name_col, self.smiles_col, 'pKa'] + [f for f in self.features if f not in [self.name_col, self.smiles_col, 'pKa']]
        tree = ttk.Treeview(win, columns=columns, show='headings', height=10)
        for col in columns:
            tree.heading(col, text=col)
        for idx, row in self.dataset.iterrows():
            tree.insert("", tk.END, values=[row.get(col, "") for col in columns])
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 分子结构独立弹窗
        def show_structure(smiles, title='Molecule Structure'):
            if self.structure_window is not None and self.structure_window.winfo_exists():
                self.structure_window.destroy()
            self.structure_window = tk.Toplevel(self.root)
            self.structure_window.title(title)
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                img_tk = ImageTk.PhotoImage(img)
                lbl = tk.Label(self.structure_window, image=img_tk)
                lbl.image = img_tk
                lbl.pack()
            else:
                tk.Label(self.structure_window, text="Invalid SMILES!").pack()

        def on_select(event):
            cur_item = tree.focus()
            if not cur_item:
                return
            v = tree.item(cur_item, 'values')
            smiles = v[columns.index(self.smiles_col)]
            show_structure(smiles, title=f"Structure: {v[columns.index(self.name_col)]}")

        tree.bind('<<TreeviewSelect>>', on_select)

        # pKa分布柱状图
        fig, ax = plt.subplots(figsize=(4,3))
        ax.hist(self.dataset['pKa'].dropna(), bins=8, color="skyblue", edgecolor="black")
        ax.set_xlabel("pKa")
        ax.set_ylabel("Count")
        ax.set_title("pKa Distribution")
        plt.tight_layout()
        plt.show()

    def train_model(self):
        """
        Train model with depth selection at the beginning
        """
        # 首先选择深度
        self.show_depth_selection()

    def show_depth_selection(self):
        """
        Show a dialog for selecting decision tree depth
        """
        depth_win = tk.Toplevel(self.root)
        depth_win.title("Decision Tree Depth Selection")
        depth_win.geometry("500x450")
        depth_win.resizable(False, False)
        
        # 创建滚动框架
        main_frame = tk.Frame(depth_win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 标题和说明
        tk.Label(scrollable_frame, text="Select Decision Tree Depth", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        explanation = """What is Tree Depth and Why Does it Matter?

• Depth controls how many sequential questions the tree can ask
• Each level adds more complexity and decision rules
• Shallow trees (depth 1-2): 
  - Simple, easy to interpret and explain
  - May miss important patterns (underfitting)
  - Good for initial understanding
• Medium trees (depth 3-4): 
  - Balanced complexity and interpretability
  - Usually optimal for most problems
  - Recommended starting point
• Deep trees (depth 5+): 
  - Very complex, capture subtle patterns
  - Risk of memorizing training data (overfitting)
  - Hard to interpret and may not generalize well

Try different depths to see how training vs validation performance changes!"""
        
        tk.Label(scrollable_frame, text=explanation, justify=tk.LEFT, 
                font=("Arial", 9)).pack(pady=10, padx=20)
        
        # 深度选择
        tk.Label(scrollable_frame, text="Choose Maximum Depth:", 
                font=("Arial", 12, "bold")).pack(pady=(20,5))
        
        depth_var = tk.IntVar(value=self.selected_depth)  # 使用之前选择的深度
        
        # 创建单选按钮
        depths = [1, 2, 3, 4, 5, 6]
        depth_descriptions = [
            "1 - Very Simple (1 split only)",
            "2 - Simple (2 levels)",
            "3 - Moderate (3 levels) - Recommended",
            "4 - Complex (4 levels)",
            "5 - Very Complex (5 levels)",
            "6 - Extremely Complex (6 levels)"
        ]
        
        for depth, desc in zip(depths, depth_descriptions):
            rb = tk.Radiobutton(scrollable_frame, text=desc, variable=depth_var, 
                            value=depth, font=("Arial", 10))
            rb.pack(anchor=tk.W, padx=40, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # 按钮框架
        btn_frame = tk.Frame(depth_win)
        btn_frame.pack(pady=20)
        
        def train_with_depth():
            self.selected_depth = depth_var.get()
            depth_win.destroy()
            self.train_model_with_depth(self.selected_depth)
        
        def cancel():
            depth_win.destroy()
        
        tk.Button(btn_frame, text="Train Model", command=train_with_depth,
                bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                padx=20, pady=5).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="Cancel", command=cancel,
                bg="#f44336", fg="white", font=("Arial", 10, "bold"),
                padx=20, pady=5).pack(side=tk.LEFT, padx=10)

    def train_model_with_depth(self, max_depth):
        """
        Train model with specified depth and show results
        """
        # 只保留完整数据，固定随机种子分训练/验证集
        df = self.dataset.dropna(subset=self.features + ['pKa'])
        X = df[self.features].values
        y = df['pKa'].values
        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_X, self.train_y = train_X, train_y
        self.val_X, self.val_y = val_X, val_y

        # 记录验证集索引行
        val_mask = np.isin(y, val_y)
        self.val_idx = df.index[val_mask]

        # 使用选择的深度训练模型
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        self.model.fit(train_X, train_y)

        # 预测
        pred_train = self.model.predict(train_X)
        pred_val = self.model.predict(val_X)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        train_mae = mean_absolute_error(train_y, pred_train)
        val_mae = mean_absolute_error(val_y, pred_val)
        train_mse = mean_squared_error(train_y, pred_train)
        val_mse = mean_squared_error(val_y, pred_val)
        train_r2 = r2_score(train_y, pred_train)
        val_r2 = r2_score(val_y, pred_val)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(train_y, pred_train, color='blue', alpha=0.7, label='Train set', s=60)
        ax.scatter(val_y, pred_val, color='orange', alpha=0.8, label='Validation set', s=60)
        minval = min(np.min(train_y), np.min(val_y), np.min(pred_train), np.min(pred_val)) - 0.5
        maxval = max(np.max(train_y), np.max(val_y), np.max(pred_train), np.max(pred_val)) + 0.5
        ax.plot([minval, maxval], [minval, maxval], 'r--', label='Perfect Prediction', linewidth=2)
        ax.set_xlabel('True pKa', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted pKa', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance (Tree Depth = {max_depth})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 标注性能指标（包含MSE）
        textstr = (f'Training MAE: {train_mae:.3f}\n'
                f'Validation MAE: {val_mae:.3f}\n'
                f'Training MSE: {train_mse:.3f}\n'
                f'Validation MSE: {val_mse:.3f}\n'
                f'Training R²: {train_r2:.3f}\n'
                f'Validation R²: {val_r2:.3f}\n\n'
                f'Tree Depth: {max_depth}')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

        # 同步弹窗展示验证集分子结构和预测值
        val_df = df.iloc[self.val_idx, :]
        val_win = tk.Toplevel(self.root)
        val_win.title("Validation Set Molecules")
        for i, row in val_df.iterrows():
            smiles = row[self.smiles_col]
            name = row[self.name_col]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(150, 150))
                img_tk = ImageTk.PhotoImage(img)
                pred_value = self.model.predict([row[self.features].values])[0]
                lbl = tk.Label(val_win, 
                            text=f"{name}\nTrue pKa={row['pKa']:.2f}\nPredicted={pred_value:.2f}\nError={abs(row['pKa']-pred_value):.2f}", 
                            image=img_tk, compound=tk.TOP)
                lbl.image = img_tk
                lbl.pack(side=tk.LEFT, padx=5)

        self.tree_btn.config(state=tk.NORMAL)
        self.predict_btn.config(state=tk.NORMAL)
        
        # 打印深度影响分析（包含MSE）
        print(f"\nModel Training Complete (Depth = {max_depth}):")
        print("-" * 60)
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        print(f"MAE Gap (Val-Train): {val_mae-train_mae:.4f}")
        print(f"MSE Gap (Val-Train): {val_mse-train_mse:.4f}")
        
        # 给出建议
        if val_mae > train_mae * 1.5:
            print("⚠️  Warning: Large gap between training and validation error suggests overfitting!")
        elif val_mae > train_mae * 1.2:
            print("💡 Note: Some overfitting detected. Consider reducing depth.")
        else:
            print("✅ Good balance between training and validation performance.")
    def show_chem_tree(self):
        """
        Display the trained decision tree with enhanced legend
        """
        if self.model is None or not self.features:
            messagebox.showerror("Error", "Please train the model first!")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.tree import plot_tree, _tree
        import pandas as pd
        import matplotlib.patches as mpatches
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # 计算性能指标
        pred_train = self.model.predict(self.train_X)
        pred_val = self.model.predict(self.val_X)
        train_mae = mean_absolute_error(self.train_y, pred_train)
        val_mae = mean_absolute_error(self.val_y, pred_val)
        val_r2 = r2_score(self.val_y, pred_val)
        
        # 绘制决策树
        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(
            self.model,
            ax=ax,
            feature_names=self.features,
            filled=True,
            impurity=False,
            rounded=True,
            fontsize=10,
            precision=2
        )
        
        tree_ = self.model.tree_
        leaf_node_ids = np.where(tree_.children_left == _tree.TREE_LEAF)[0]
        
        # 增强的决策树概念说明（30%更多内容）
        legend_text = (
            f"How to Read This Tree:\n"
            f"• Start at the top (root node)\n"
            f"• Each box shows a yes/no question\n"
            f"• Follow the path based on your molecule's features\n"
            f"• Darker colors = higher pKa predictions\n"
            f"• Each leaf gives a final pKa prediction\n"
            f"• Each path corresponds to a set of feature-based rules\n"
        )
        
        # 创建图例
        dummy_patch = mpatches.Patch(color='white', alpha=0.0, label=legend_text)
        ax.legend(
            handles=[dummy_patch],
            loc='upper left',
            fontsize=9,
            handlelength=0,
            handletextpad=0,
            frameon=True,
            fancybox=True,
            borderpad=1.2
        )
        
        # 添加标题
        plt.suptitle(f"Chemical Decision Tree (Depth = {self.selected_depth})", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.25, right=0.95)
        
        # 保存图片
        filename = f"decision_tree_depth_{self.selected_depth}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Decision tree figure saved as: {filename}")
        plt.show()
        
        # 显示特征重要性
        self.show_feature_importance()

    def show_feature_importance(self):
        """
        Display feature importance plot for the trained model
        """
        if self.model is None:
            return
            
        # 获取特征重要性
        importances = self.model.feature_importances_
        
        # 创建特征重要性图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按重要性排序
        indices = np.argsort(importances)[::-1]
        sorted_features = [self.features[i] for i in indices]
        sorted_importances = importances[indices]
        
        # 绘制条形图
        bars = ax.bar(range(len(sorted_features)), sorted_importances, 
                    color=['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#8B4513', 
                            '#9932CC', '#DC143C', '#FF6347', '#32CD32'][:len(sorted_features)])
        
        # 清理特征名称，去掉括号及其内容
        clean_features = []
        for feature in sorted_features:
            clean_feature = feature.split('(')[0].strip()
            clean_features.append(clean_feature)
        
        # 设置标签和标题
        ax.set_xlabel('Molecular Descriptors', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Importance (Tree Depth = {self.selected_depth})', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(sorted_features)))
        ax.set_xticklabels(clean_features, rotation=45, ha='right')
        
        # 在条形图上添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, sorted_importances)):
            height = bar.get_height()
            if height > 0.001:  # 只显示有意义的值
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 添加网格
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(sorted_importances) * 1.2 if max(sorted_importances) > 0 else 1)
        
        # 添加解释文本（移到右上角）
        explanation_text = (
            f"Feature Importance (Depth {self.selected_depth}):\n"
            "• Higher values = more important features\n"
            "• Sum of all importance values = 1.0\n"
            "• Shallow trees use fewer features\n"
            "• Importance = how much each feature reduces prediction error"
        )
        
        ax.text(0.98, 0.98, explanation_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        filename = f"feature_importance_depth_{self.selected_depth}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Feature importance figure saved as: {filename}")
        plt.show()

    def predict(self):
        # 只允许输入特征，显示预测
        predict_win = tk.Toplevel(self.root)
        predict_win.title("Predict pKa")
        predict_win.geometry("500x600")
        
        # 创建滚动框架
        main_frame = tk.Frame(predict_win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 允许同学填写特征，选择测试集分子
        test_choices = []
        if self.testset is not None:
            for _, row in self.testset.iterrows():
                test_choices.append(f"{row[self.name_col]} ({row[self.smiles_col]})")
        var = tk.StringVar(value=test_choices[0] if test_choices else "")

        def on_select(event=None):
            if test_choices:
                idx = test_choices.index(var.get())
                smiles = self.testset.iloc[idx][self.smiles_col]
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(200, 200))
                    img_tk = ImageTk.PhotoImage(img)
                    img_lbl.configure(image=img_tk)
                    img_lbl.image = img_tk

        tk.Label(scrollable_frame, text="Select a molecule:").pack()
        combo = ttk.Combobox(scrollable_frame, values=test_choices, textvariable=var, state='readonly')
        combo.pack()
        combo.bind('<<ComboboxSelected>>', on_select)
        img_lbl = tk.Label(scrollable_frame)
        img_lbl.pack()
        if test_choices:
            on_select()

        # 特征输入
        entries = {}
        for feat in self.features:
            frm = tk.Frame(scrollable_frame)
            frm.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(frm, text=feat, width=25, anchor="w").pack(side=tk.LEFT)
            ent = tk.Entry(frm, width=15)
            ent.pack(side=tk.LEFT, padx=5)
            entries[feat] = ent

        def submit():
            values = []
            for feat in self.features:
                val = entries[feat].get()
                try:
                    values.append(float(val))
                except:
                    messagebox.showerror("Error", f"Invalid value for {feat}")
                    return
            pred = self.model.predict([values])[0]
            messagebox.showinfo("Prediction", f"Predicted pKa: {pred:.2f}\n(Using tree depth {self.selected_depth})")

        tk.Button(scrollable_frame, text="Submit", command=submit).pack(pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()