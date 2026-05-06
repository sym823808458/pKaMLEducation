import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageTk
from sklearn.tree import DecisionTreeRegressor, plot_tree, _tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive pKa Prediction Tool")
        self.root.geometry("480x600")

        self.full_df = None
        self.pool_df = None
        self.test_df = None
        self.selected_indices = set()

        self.base_features = []
        self.custom_col_names = []

        self.model = None
        self.selected_depth = 3
        self._valid_features = []

        self.train_df = None
        self.val_df = None
        self.train_X = self.train_y = None
        self.val_X = self.val_y = None

        self._stored_train_pool_idx = []
        self._stored_val_pool_idx = []

        self.smiles_col = "SMILES"
        self.name_col = "Molecule Name"

        self.test_flag_col = "IsTest"
        self.test_names = [
            "Octanoic acid",
            "3-Chlorobutanoic acid",
            "Iodoacetic acid",
            "Phenylglyoxylic acid",
        ]

        self.structure_window = None
        self._current_edit_entry = None

        self.feature_hints = {
            "Carbon": "Count all carbon atoms in the molecule",
            "COOH": "Count the carboxyl (-COOH) groups",
            "Halogen": "Count halogen atoms (F, Cl, Br, I)",
            "OH": "Count hydroxyl (-OH) groups (not those in -COOH)",
            "NO2": "Count nitro (-NO₂) groups",
            "EWG_Flag": "EWG → +1,  EDG → -1,  None → 0",
            "EWG_Pos": "α-position → 1,  β-position → 2,  γ-position → 3,  None → 0",
            "EWG_Rank": "None=0, OH=1, I=2, Br=3, Cl=4, F=6, NO₂=7",
        }

        self.create_widgets()

    # ──────────────────── property ────────────────────
    @property
    def features(self):
        return self.base_features + self.custom_col_names

    # ══════════════════════════════════════════════════
    #                   CREATE WIDGETS
    # ══════════════════════════════════════════════════
    def create_widgets(self):
        tk.Label(self.root, text="Interactive pKa Prediction",
                 font=("Times New Roman", 18, "bold")).pack(pady=15)

        btn_cfg = dict(font=("Times New Roman", 13), width=32)

        self.upload_btn = tk.Button(self.root, text="Step 1 : Upload CSV",
                                    command=self.upload_file, **btn_cfg)
        self.upload_btn.pack(pady=6)

        self.manage_btn = tk.Button(self.root, text="Step 2 : Data Management",
                                    command=self.manage_data,
                                    state=tk.DISABLED, **btn_cfg)
        self.manage_btn.pack(pady=6)

        self.view_btn = tk.Button(self.root, text="Step 3 : Visualize & Split Data",
                                  command=self.visualize_dataset,
                                  state=tk.DISABLED, **btn_cfg)
        self.view_btn.pack(pady=6)

        self.train_btn = tk.Button(self.root, text="Step 4 : Train Model",
                                   command=self.train_model,
                                   state=tk.DISABLED, **btn_cfg)
        self.train_btn.pack(pady=6)

        self.tree_btn = tk.Button(self.root, text="Step 5 : Decision Tree & Importance",
                                  command=self.show_chem_tree,
                                  state=tk.DISABLED, **btn_cfg)
        self.tree_btn.pack(pady=6)

        self.predict_btn = tk.Button(self.root, text="Step 6 : Predict Test Set",
                                     command=self.predict,
                                     state=tk.DISABLED, **btn_cfg)
        self.predict_btn.pack(pady=6)

        self.status_var = tk.StringVar(value="Please upload a CSV file to begin.")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Times New Roman", 10), fg="gray",
                 wraplength=450, justify=tk.CENTER).pack(pady=12)

    # ══════════════════════════════════════════════════
    #            STEP 1 : Upload CSV
    # ══════════════════════════════════════════════════
    def upload_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        df = pd.read_csv(path)
        self.full_df = df.copy()

        names_low = df[self.name_col].str.strip().str.lower()
        test_low = [n.strip().lower() for n in self.test_names]
        test_mask = names_low.isin(test_low)

        df[self.test_flag_col] = 0
        df.loc[test_mask, self.test_flag_col] = 1

        self.test_df = df[test_mask].reset_index(drop=True)
        self.pool_df = df[~test_mask].reset_index(drop=True)

        exclude = {self.smiles_col, self.name_col, "pKa", "Split", self.test_flag_col}
        self.base_features = [c for c in df.columns if c not in exclude]

        self.selected_indices = set()
        self.custom_col_names = []
        self.model = None
        self._valid_features = []
        self._stored_train_pool_idx = []
        self._stored_val_pool_idx = []

        self.manage_btn.config(state=tk.NORMAL)
        self.view_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.tree_btn.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.DISABLED)

        n_t = len(self.test_df)
        n_p = len(self.pool_df)
        self.status_var.set(
            f"Loaded {len(df)} molecules: {n_p} available, {n_t} test set")

        found = self.test_df[self.name_col].tolist()
        msg = (f"Dataset loaded!\n\n"
               f"Total molecules: {len(df)}\n"
               f"Available for training: {n_p}\n"
               f"Fixed test set ({n_t}):\n")
        for nm in found:
            msg += f"  • {nm}\n"
        if n_t < len(self.test_names):
            expected = {n.strip().lower() for n in self.test_names}
            got = {n.strip().lower() for n in found}
            missing = expected - got
            msg += f"\n⚠️  Not found in CSV: {missing}"

        messagebox.showinfo("Dataset Info", msg)

    # ══════════════════════════════════════════════════
    #         STEP 2 : Data Management (Unified Table)
    # ══════════════════════════════════════════════════
    def manage_data(self):
        win = tk.Toplevel(self.root)
        win.title("Step 2 : Data Management – Molecule & Feature Editor")
        win.geometry("1400x800")

        tk.Label(win,
                 text=("Double-click any cell to edit  |  "
                       "Add/Remove molecules using buttons below  |  "
                       "Pink rows = TEST set (can be toggled)"),
                 font=("Times New Roman", 11, "italic"), fg="blue"
                 ).pack(fill=tk.X, padx=10, pady=5)

        bf = tk.Frame(win)
        bf.pack(fill=tk.X, padx=10, pady=5)

        tree_ref = {'tree': None}

        def add_molecule():
            dlg = tk.Toplevel(win)
            dlg.title("Add New Molecule")
            dlg.geometry("500x420")
            dlg.transient(win)
            dlg.grab_set()

            tk.Label(dlg, text="Add New Molecule",
                     font=("Times New Roman", 14, "bold")).pack(pady=10)

            tk.Label(dlg, text="Molecule Name:",
                     font=("Times New Roman", 11)).pack(anchor="w", padx=20)
            name_entry = tk.Entry(dlg, font=("Times New Roman", 11), width=45)
            name_entry.pack(padx=20, pady=2)
            name_entry.focus()

            tk.Label(dlg, text="SMILES:",
                     font=("Times New Roman", 11)).pack(anchor="w", padx=20)
            smiles_entry = tk.Entry(dlg, font=("Times New Roman", 11), width=45)
            smiles_entry.pack(padx=20, pady=2)

            tk.Label(dlg, text="pKa (required):",
                     font=("Times New Roman", 11)).pack(anchor="w", padx=20)
            pka_entry = tk.Entry(dlg, font=("Times New Roman", 11), width=45)
            pka_entry.pack(padx=20, pady=2)

            tk.Label(dlg, text="Feature Values:",
                     font=("Times New Roman", 11)).pack(anchor="w", padx=20, pady=(10, 2))
            feat_entries = {}
            feat_frame = tk.Frame(dlg)
            feat_frame.pack(padx=20, pady=5)
            for i, feat in enumerate(self.features):
                tk.Label(feat_frame, text=f"{feat}:",
                         font=("Times New Roman", 10)).grid(row=i//2, column=(i%2)*3, sticky="w")
                e = tk.Entry(feat_frame, font=("Times New Roman", 10), width=12)
                e.grid(row=i//2, column=(i%2)*3+1, padx=5, pady=2)
                feat_entries[feat] = e

            istest_var = tk.BooleanVar(value=False)
            tk.Checkbutton(dlg, text="Mark as Test Set Molecule",
                           variable=istest_var,
                           font=("Times New Roman", 11)).pack(pady=10)

            preview_frame = tk.Frame(dlg)
            preview_frame.pack(pady=5)
            preview_label = tk.Label(preview_frame, text="SMILES Preview",
                                    font=("Times New Roman", 10, "italic"))
            preview_label.pack()
            preview_img_label = tk.Label(preview_frame)

            def update_preview():
                smiles = smiles_entry.get().strip()
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        img = Draw.MolToImage(mol, size=(200, 150))
                        itk = ImageTk.PhotoImage(img)
                        preview_img_label.config(image=itk)
                        preview_img_label.image = itk
                        preview_img_label.pack()
                    else:
                        preview_img_label.config(text="Invalid SMILES", fg="red")
                else:
                    preview_img_label.config(text="", fg="black")

            smiles_entry.bind("<KeyRelease>", lambda e: update_preview())

            def do_save():
                name = name_entry.get().strip()
                smiles = smiles_entry.get().strip()
                pka_str = pka_entry.get().strip()

                if not name:
                    messagebox.showwarning("Warning", "Please enter a molecule name!", parent=dlg)
                    return

                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    messagebox.showwarning("Warning",
                                         "Invalid SMILES! Please check and try again.",
                                         parent=dlg)
                    return

                if not pka_str:
                    messagebox.showwarning("Warning", "Please enter a pKa value (required)!", parent=dlg)
                    return
                try:
                    pka_val = float(pka_str)
                except ValueError:
                    messagebox.showwarning("Warning", "pKa must be a number!", parent=dlg)
                    return

                existing_names = set(self.pool_df[self.name_col].tolist() +
                                   self.test_df[self.name_col].tolist())
                if name.lower() in {n.lower() for n in existing_names}:
                    messagebox.showwarning("Warning",
                                         f"A molecule named '{name}' already exists!",
                                         parent=dlg)
                    return

                feat_vals = {}
                for feat, entry in feat_entries.items():
                    val = entry.get().strip()
                    feat_vals[feat] = float(val) if val else np.nan

                new_row = {self.name_col: name, self.smiles_col: smiles, 'pKa': pka_val,
                          self.test_flag_col: 1 if istest_var.get() else 0}
                new_row.update(feat_vals)

                if istest_var.get():
                    self.test_df = pd.concat([self.test_df, pd.DataFrame([new_row])],
                                            ignore_index=True)
                else:
                    self.pool_df = pd.concat([self.pool_df, pd.DataFrame([new_row])],
                                            ignore_index=True)
                    self.selected_indices.add(len(self.pool_df) - 1)

                dlg.destroy()
                refresh_table()

            tk.Button(dlg, text="Add Molecule", command=do_save,
                      font=("Times New Roman", 11, "bold"),
                      bg="#4CAF50", fg="white", padx=20, pady=5).pack(pady=10)

        def remove_molecule():
            tree = tree_ref['tree']
            if tree is None:
                return

            selection = tree.selection()
            if not selection:
                messagebox.showinfo("Info", "Please select a molecule to remove.", parent=win)
                return

            if not messagebox.askyesno("Confirm",
                                      "Remove selected molecule(s)?",
                                      parent=win):
                return

            pool_indices_to_delete = []
            test_indices_to_delete = []

            for item in selection:
                if item == "sep":
                    continue
                if item.startswith("pool_"):
                    idx = int(item.split("_")[1])
                    pool_indices_to_delete.append(idx)
                elif item.startswith("test_"):
                    idx = int(item.split("_")[1])
                    test_indices_to_delete.append(idx)

            pool_indices_to_delete.sort(reverse=True)
            test_indices_to_delete.sort(reverse=True)

            for idx in pool_indices_to_delete:
                self.pool_df = self.pool_df.drop(idx).reset_index(drop=True)

            for idx in test_indices_to_delete:
                self.test_df = self.test_df.drop(idx).reset_index(drop=True)

            self.selected_indices = set(range(len(self.pool_df)))
            refresh_table()

        def add_column():
            dlg = tk.Toplevel(win)
            dlg.title("Add Feature Column")
            dlg.geometry("380x160")
            dlg.transient(win)
            dlg.grab_set()
            tk.Label(dlg, text="Enter new feature column name:",
                     font=("Times New Roman", 11)).pack(pady=10)
            ne = tk.Entry(dlg, font=("Times New Roman", 11), width=32)
            ne.pack(pady=5)
            ne.focus()

            def do_add():
                cn = ne.get().strip()
                if not cn:
                    return
                all_existing = (self.base_features + self.custom_col_names
                                + [self.smiles_col, self.name_col, "pKa", self.test_flag_col])
                if cn in all_existing:
                    messagebox.showwarning("Warning",
                                           f"'{cn}' already exists!", parent=dlg)
                    return
                self.pool_df[cn] = np.nan
                self.test_df[cn] = np.nan
                self.custom_col_names.append(cn)
                dlg.destroy()
                refresh_table()

            ne.bind("<Return>", lambda e: do_add())
            tk.Button(dlg, text="Add", command=do_add,
                      font=("Times New Roman", 10, "bold"),
                      bg="#4CAF50", fg="white").pack(pady=10)

        def remove_column():
            if not self.custom_col_names:
                messagebox.showinfo("Info",
                                    "No custom columns to remove.", parent=win)
                return
            dlg = tk.Toplevel(win)
            dlg.title("Remove Feature Column")
            dlg.geometry("350x160")
            dlg.transient(win)
            dlg.grab_set()
            tk.Label(dlg, text="Select column to remove:",
                     font=("Times New Roman", 11)).pack(pady=10)
            cv = tk.StringVar(value=self.custom_col_names[0])
            ttk.Combobox(dlg, values=self.custom_col_names,
                         textvariable=cv, state="readonly").pack(pady=5)

            def do_rm():
                c = cv.get()
                if c in self.custom_col_names:
                    self.custom_col_names.remove(c)
                    self.pool_df.drop(columns=[c], inplace=True, errors='ignore')
                    self.test_df.drop(columns=[c], inplace=True, errors='ignore')
                dlg.destroy()
                refresh_table()

            tk.Button(dlg, text="Remove", command=do_rm,
                      font=("Times New Roman", 10, "bold"),
                      bg="#f44336", fg="white").pack(pady=10)

        tk.Button(bf, text="＋ Add Molecule", command=add_molecule,
                  font=("Times New Roman", 10, "bold"),
                  bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="－ Remove Selected", command=remove_molecule,
                  font=("Times New Roman", 10, "bold"),
                  bg="#f44336", fg="white").pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="＋ Add Feature Column", command=add_column,
                  font=("Times New Roman", 10, "bold"),
                  bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="－ Remove Feature Column", command=remove_column,
                  font=("Times New Roman", 10, "bold"),
                  bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=4)

        info_var = tk.StringVar(value="")
        tk.Label(bf, textvariable=info_var,
                 font=("Times New Roman", 10), fg="purple").pack(side=tk.RIGHT, padx=5)

        table_box = tk.Frame(win)
        table_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def refresh_table():
            for w in table_box.winfo_children():
                w.destroy()

            all_f = self.base_features + self.custom_col_names
            cols = ["Type", self.name_col, "SMILES", "pKa", "Test"] + all_f

            tree = ttk.Treeview(table_box, columns=cols,
                                show="headings", height=30, selectmode="extended")
            tree_ref['tree'] = tree
            tree.column("Type", width=50, minwidth=50)
            tree.column(self.name_col, width=150, minwidth=100)
            tree.column("SMILES", width=180, minwidth=120)
            tree.column("pKa", width=60, minwidth=50)
            tree.column("Test", width=50, minwidth=50)
            for c in all_f:
                tree.column(c, width=80, minwidth=50)
            for c in cols:
                tree.heading(c, text=c)

            tree.tag_configure("pool", background="#F0FFF0")
            tree.tag_configure("test", background="#FFE4E1")

            for idx, row in self.pool_df.iterrows():
                vals = ["Pool", row[self.name_col], row.get(self.smiles_col, ""),
                        f"{row['pKa']:.2f}" if pd.notna(row.get('pKa')) else "",
                        ""]
                for f in all_f:
                    v = row.get(f)
                    vals.append(f"{v}" if pd.notna(v) else "")
                tree.insert("", tk.END, iid=f"pool_{idx}",
                            values=vals, tags=("pool",))

            tree.insert("", tk.END, iid="sep",
                        values=["── ──"] + ["── ──"] * (len(cols) - 1))

            for idx, row in self.test_df.iterrows():
                vals = ["TEST", row[self.name_col], row.get(self.smiles_col, ""),
                        f"{row['pKa']:.2f}" if pd.notna(row.get('pKa')) else "",
                        "✓"]
                for f in all_f:
                    v = row.get(f)
                    vals.append(f"{v}" if pd.notna(v) else "")
                tree.insert("", tk.END, iid=f"test_{idx}",
                            values=vals, tags=("test",))

            vs = ttk.Scrollbar(table_box, orient="vertical", command=tree.yview)
            hs = ttk.Scrollbar(table_box, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
            tree.grid(row=0, column=0, sticky="nsew")
            vs.grid(row=0, column=1, sticky="ns")
            hs.grid(row=1, column=0, sticky="ew")
            table_box.grid_rowconfigure(0, weight=1)
            table_box.grid_columnconfigure(0, weight=1)

            info_var.set(f"Pool: {len(self.pool_df)}  |  "
                         f"Test: {len(self.test_df)}  |  "
                         f"Features: {len(all_f)} "
                         f"({len(self.base_features)} base + "
                         f"{len(self.custom_col_names)} custom)")

            def on_dbl(event):
                if self._current_edit_entry:
                    try:
                        self._current_edit_entry.destroy()
                    except Exception:
                        pass
                    self._current_edit_entry = None

                region = tree.identify("region", event.x, event.y)
                if region != "cell":
                    return
                col_id = tree.identify_column(event.x)
                col_num = int(col_id[1:]) - 1
                if col_num == 0:
                    return
                item = tree.identify_row(event.y)
                if not item or item == "sep":
                    return
                bbox = tree.bbox(item, col_id)
                if not bbox:
                    return

                cur_vals = list(tree.item(item, "values"))
                cur_val = cur_vals[col_num] if col_num < len(cur_vals) else ""

                entry = tk.Entry(tree, font=("Times New Roman", 10))
                entry.place(x=bbox[0], y=bbox[1],
                            width=bbox[2], height=bbox[3])
                entry.insert(0, cur_val)
                entry.focus()
                entry.select_range(0, tk.END)
                self._current_edit_entry = entry

                def save(event=None):
                    nv = entry.get().strip()

                    if col_num == 2:
                        if nv:
                            mol = Chem.MolFromSmiles(nv)
                            if not mol:
                                messagebox.showwarning("Warning",
                                                     "Invalid SMILES! Please check and try again.",
                                                     parent=win)
                                entry.focus()
                                return

                    entry.destroy()
                    self._current_edit_entry = None

                    if item.startswith("pool_"):
                        ix = int(item.split("_")[1])
                        if col_num == 1:
                            self.pool_df.at[ix, self.name_col] = nv
                        elif col_num == 2:
                            self.pool_df.at[ix, self.smiles_col] = nv
                        elif col_num == 3:
                            self.pool_df.at[ix, 'pKa'] = float(nv) if nv else np.nan
                        elif col_num == 4:
                            pass
                        else:
                            feat_idx = col_num - 5
                            if feat_idx < len(all_f):
                                feat_name = all_f[feat_idx]
                                self.pool_df.at[ix, feat_name] = float(nv) if nv else np.nan
                    elif item.startswith("test_"):
                        ix = int(item.split("_")[1])
                        if col_num == 1:
                            self.test_df.at[ix, self.name_col] = nv
                        elif col_num == 2:
                            self.test_df.at[ix, self.smiles_col] = nv
                        elif col_num == 3:
                            self.test_df.at[ix, 'pKa'] = float(nv) if nv else np.nan
                        elif col_num == 4:
                            pass
                        else:
                            feat_idx = col_num - 5
                            if feat_idx < len(all_f):
                                feat_name = all_f[feat_idx]
                                self.test_df.at[ix, feat_name] = float(nv) if nv else np.nan

                    cur_vals[col_num] = nv
                    tree.item(item, values=cur_vals)

                entry.bind("<Return>", save)
                entry.bind("<FocusOut>", save)
                entry.bind("<Escape>", lambda e: (
                    entry.destroy(),
                    setattr(self, '_current_edit_entry', None)))

            tree.bind("<Double-1>", on_dbl)

        refresh_table()

        bot = tk.Frame(win)
        bot.pack(fill=tk.X, padx=10, pady=10)

        def save_close():
            self.selected_indices = set(range(len(self.pool_df)))
            if len(self.selected_indices) == 0:
                messagebox.showwarning("Warning",
                                       "Please add at least some molecules!",
                                       parent=win)
                return
            self.status_var.set(
                f"Selected {len(self.selected_indices)} molecules  |  "
                f"Features: {len(self.features)} "
                f"({len(self.base_features)} base + "
                f"{len(self.custom_col_names)} custom)")
            self.view_btn.config(state=tk.NORMAL)
            self.train_btn.config(state=tk.DISABLED)
            self.tree_btn.config(state=tk.DISABLED)
            self.predict_btn.config(state=tk.DISABLED)
            self._stored_train_pool_idx = []
            self._stored_val_pool_idx = []
            self.model = None
            win.destroy()

        tk.Button(bot, text="Save & Continue", command=save_close,
                  font=("Times New Roman", 13, "bold"),
                  bg="#2196F3", fg="white",
                  padx=24, pady=6).pack()

    # ══════════════════════════════════════════════════
    #  STEP 3 : Visualize & Auto-Split
    # ══════════════════════════════════════════════════
    def visualize_dataset(self):
        if not self.selected_indices:
            messagebox.showwarning("Warning",
                                   "No molecules selected. Open Step 2 first.")
            return

        # ── FIX: close any lingering matplotlib figures from previous steps ──
        plt.close('all')

        sel_list = sorted(self.selected_indices)
        sel = self.pool_df.loc[sel_list]
        n = len(sel)

        if n < 3:
            messagebox.showerror("Error",
                                 f"Only {n} molecules selected. Need at least 3.")
            return

        vs = max(1, int(n * 0.2))
        vs = min(vs, n - 2)

        positions = np.arange(n)
        train_pos, val_pos = train_test_split(
            positions, test_size=vs, random_state=42)

        self._stored_train_pool_idx = [sel_list[p] for p in train_pos]
        self._stored_val_pool_idx = [sel_list[p] for p in val_pos]

        train_sel = sel.iloc[train_pos]
        val_sel = sel.iloc[val_pos]

        sw = tk.Toplevel(self.root)
        sw.title("Step 3 : Train / Validation Split & Visualization")
        sw.geometry("1100x750")

        canvas = tk.Canvas(sw)
        sb = ttk.Scrollbar(sw, orient="vertical", command=canvas.yview)
        sf = tk.Frame(canvas)
        sf.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        tk.Label(sf,
                 text=(f"Auto Split :  {len(train_sel)} Train (80%)  +  "
                       f"{len(val_sel)} Validation (20%)"),
                 font=("Times New Roman", 14, "bold"), fg="purple"
                 ).pack(pady=6)

        tk.Label(sf,
                 text=f"Training Set : {len(train_sel)} molecules",
                 font=("Times New Roman", 15, "bold"), fg="#2E7D32"
                 ).pack(pady=8)

        mpr = 5
        rf = None
        for i, (_, row) in enumerate(train_sel.iterrows()):
            if i % mpr == 0:
                rf = tk.Frame(sf)
                rf.pack(fill=tk.X, padx=5, pady=4)
            mol = Chem.MolFromSmiles(row[self.smiles_col])
            if mol:
                img = Draw.MolToImage(mol, size=(180, 180))
                itk = ImageTk.PhotoImage(img)
                pka_s = (f"{row['pKa']:.2f}"
                         if pd.notna(row.get('pKa')) else "?")
                mf = tk.Frame(rf, bd=2, relief=tk.GROOVE, bg="#E8F5E9")
                mf.pack(side=tk.LEFT, padx=3, pady=3)
                lbl = tk.Label(mf, image=itk, bg="#E8F5E9")
                lbl.image = itk
                lbl.pack()
                tk.Label(mf,
                         text=f"{row[self.name_col]}\npKa = {pka_s}",
                         font=("Times New Roman", 9),
                         justify=tk.CENTER, bg="#E8F5E9").pack()

        tk.Label(sf,
                 text=f"\nValidation Set : {len(val_sel)} molecules",
                 font=("Times New Roman", 15, "bold"), fg="#E65100"
                 ).pack(pady=8)

        rf = None
        for i, (_, row) in enumerate(val_sel.iterrows()):
            if i % mpr == 0:
                rf = tk.Frame(sf)
                rf.pack(fill=tk.X, padx=5, pady=4)
            mol = Chem.MolFromSmiles(row[self.smiles_col])
            if mol:
                img = Draw.MolToImage(mol, size=(180, 180))
                itk = ImageTk.PhotoImage(img)
                pka_s = (f"{row['pKa']:.2f}"
                         if pd.notna(row.get('pKa')) else "?")
                mf = tk.Frame(rf, bd=2, relief=tk.GROOVE, bg="#FFF3E0")
                mf.pack(side=tk.LEFT, padx=3, pady=3)
                lbl = tk.Label(mf, image=itk, bg="#FFF3E0")
                lbl.image = itk
                lbl.pack()
                tk.Label(mf,
                         text=f"{row[self.name_col]}\npKa = {pka_s}",
                         font=("Times New Roman", 9),
                         justify=tk.CENTER, bg="#FFF3E0").pack()

        tk.Label(sf,
                 text=f"\nFixed Test Set : {len(self.test_df)} molecules",
                 font=("Times New Roman", 15, "bold"), fg="red").pack(pady=8)
        trf = tk.Frame(sf)
        trf.pack(fill=tk.X, padx=5, pady=5)
        for _, row in self.test_df.iterrows():
            mol = Chem.MolFromSmiles(row[self.smiles_col])
            if mol:
                img = Draw.MolToImage(mol, size=(210, 210))
                itk = ImageTk.PhotoImage(img)
                pka_s = (f"{row['pKa']:.2f}"
                         if pd.notna(row.get('pKa')) else "?")
                mf = tk.Frame(trf, bd=3, relief=tk.RIDGE, bg="#FFE4E1")
                mf.pack(side=tk.LEFT, padx=8, pady=5)
                lbl = tk.Label(mf, image=itk, bg="#FFE4E1")
                lbl.image = itk
                lbl.pack()
                tk.Label(mf,
                         text=f"{row[self.name_col]}\npKa = {pka_s}",
                         font=("Times New Roman", 10, "bold"),
                         justify=tk.CENTER, fg="red",
                         bg="#FFE4E1").pack()

        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def _mw(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<Enter>",
                    lambda e: canvas.bind_all("<MouseWheel>", _mw))
        canvas.bind("<Leave>",
                    lambda e: canvas.unbind_all("<MouseWheel>"))

        # ── pKa distribution ──
        fig, ax = plt.subplots(figsize=(9, 6))
        all_pka = pd.concat([train_sel['pKa'].dropna(),
                             val_sel['pKa'].dropna()])
        bins = np.histogram_bin_edges(all_pka, bins=8)

        ax.hist(train_sel['pKa'].dropna(), bins=bins,
                color="skyblue", edgecolor="black", linewidth=1.5,
                alpha=0.7, label=f'Train ({len(train_sel)})')
        ax.hist(val_sel['pKa'].dropna(), bins=bins,
                color="orange", edgecolor="black", linewidth=1.5,
                alpha=0.7, label=f'Validation ({len(val_sel)})')

        ax.set_xlabel("pKa", fontsize=24, fontweight='bold',
                      fontfamily='Times New Roman')
        ax.set_ylabel("Count", fontsize=24, fontweight='bold',
                      fontfamily='Times New Roman')
        ax.set_title("pKa Distribution  (Train / Validation Split)",
                     fontsize=24, fontweight='bold',
                     fontfamily='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=30, width=2)
        for l in ax.get_xticklabels() + ax.get_yticklabels():
            l.set_fontfamily('Times New Roman')
            l.set_fontweight('bold')
        leg = ax.legend(fontsize=28, frameon=True, fancybox=True, shadow=True)
        for t in leg.get_texts():
            t.set_fontfamily('Times New Roman')
            t.set_fontweight('bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        self.train_btn.config(state=tk.NORMAL)
        self.status_var.set(
            f"Split done: {len(train_sel)} train + {len(val_sel)} val  |  "
            f"Ready to train")

    # ══════════════════════════════════════════════════
    #           STEP 4 : Train Model
    # ══════════════════════════════════════════════════
    def train_model(self):
        if not self.selected_indices:
            messagebox.showwarning("Warning",
                                   "No molecules selected. Open Step 2 first.")
            return
        if not self._stored_train_pool_idx:
            messagebox.showwarning("Warning",
                                   "Please run Step 3 first to split the data.")
            return
        self._show_depth_dialog()

    def _show_depth_dialog(self):
        dw = tk.Toplevel(self.root)
        dw.title("Decision Tree Depth Selection")
        dw.geometry("500x480")
        dw.resizable(False, False)

        mf = tk.Frame(dw)
        mf.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        c = tk.Canvas(mf)
        sb = ttk.Scrollbar(mf, orient="vertical", command=c.yview)
        sf = tk.Frame(c)
        sf.bind("<Configure>",
                lambda e: c.configure(scrollregion=c.bbox("all")))
        c.create_window((0, 0), window=sf, anchor="nw")
        c.configure(yscrollcommand=sb.set)

        tk.Label(sf, text="Select Decision Tree Depth",
                 font=("Times New Roman", 14, "bold")).pack(pady=10)

        exp = ("What is Tree Depth?\n"
               "• Depth controls how many sequential questions the tree asks\n"
               "• Shallow (1-2) : simple, easy to interpret\n"
               "• Medium (3-4) : balanced – recommended\n"
               "• Deep (5+) : complex, risk of overfitting\n\n"
               "Try different depths and observe how performance changes!")
        tk.Label(sf, text=exp, justify=tk.LEFT,
                 font=("Times New Roman", 10)).pack(pady=10, padx=20)

        tk.Label(sf, text="Choose Maximum Depth:",
                 font=("Times New Roman", 12, "bold")).pack(pady=(10, 5))
        dv = tk.IntVar(value=self.selected_depth)
        for d, desc in [(1, "1 – Very Simple"),
                        (2, "2 – Simple"),
                        (3, "3 – Moderate (Recommended)"),
                        (4, "4 – Complex"),
                        (5, "5 – Very Complex"),
                        (6, "6 – Extremely Complex")]:
            tk.Radiobutton(sf, text=desc, variable=dv, value=d,
                           font=("Times New Roman", 10)
                           ).pack(anchor=tk.W, padx=40, pady=2)

        c.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        bf = tk.Frame(dw)
        bf.pack(pady=12)

        def go():
            self.selected_depth = dv.get()
            dw.destroy()
            self._train_with_depth(self.selected_depth)

        tk.Button(bf, text="Train Model", command=go,
                  bg="#4CAF50", fg="white",
                  font=("Times New Roman", 11, "bold"),
                  padx=20, pady=5).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Cancel", command=dw.destroy,
                  bg="#f44336", fg="white",
                  font=("Times New Roman", 11, "bold"),
                  padx=20, pady=5).pack(side=tk.LEFT, padx=10)

    def _train_with_depth(self, max_depth):
        train_base = self.pool_df.loc[self._stored_train_pool_idx].copy()
        val_base = self.pool_df.loc[self._stored_val_pool_idx].copy()

        all_f = self.features
        combined = pd.concat([train_base, val_base])
        valid_f = [f for f in all_f
                   if f in combined.columns and combined[f].notna().sum() > 0]
        if not valid_f:
            messagebox.showerror("Error",
                                 "No valid features. Check your data / feature columns.")
            return

        train_clean = train_base.dropna(subset=valid_f + ['pKa'])
        val_clean = val_base.dropna(subset=valid_f + ['pKa'])

        n_excl_tr = len(train_base) - len(train_clean)
        n_excl_va = len(val_base) - len(val_clean)

        if len(train_clean) < 2:
            messagebox.showerror(
                "Error",
                f"Only {len(train_clean)} complete training rows (need ≥ 2).\n"
                f"{n_excl_tr} rows excluded due to missing values.")
            return

        if n_excl_tr + n_excl_va > 0:
            messagebox.showinfo(
                "Note",
                f"{n_excl_tr} train + {n_excl_va} val molecule(s) excluded "
                f"(missing feature values).\n"
                f"Training with {len(train_clean)} train, "
                f"{len(val_clean)} val.")

        train_X = train_clean[valid_f].values.astype(float)
        train_y = train_clean['pKa'].values.astype(float)

        has_val = len(val_clean) > 0
        if has_val:
            val_X = val_clean[valid_f].values.astype(float)
            val_y = val_clean['pKa'].values.astype(float)
        else:
            val_X = np.array([]).reshape(0, len(valid_f))
            val_y = np.array([])

        self.train_X, self.train_y = train_X, train_y
        self.val_X, self.val_y = val_X, val_y
        self.train_df = train_clean
        self.val_df = val_clean
        self._valid_features = valid_f

        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        self.model.fit(train_X, train_y)

        pred_tr = self.model.predict(train_X)
        pred_va = self.model.predict(val_X) if has_val else np.array([])

        tr_mae = mean_absolute_error(train_y, pred_tr)
        tr_mse = mean_squared_error(train_y, pred_tr)
        tr_r2 = r2_score(train_y, pred_tr) if len(train_y) > 1 else 0

        if has_val and len(val_y) > 0:
            va_mae = mean_absolute_error(val_y, pred_va)
            va_mse = mean_squared_error(val_y, pred_va)
            va_r2 = r2_score(val_y, pred_va) if len(val_y) > 1 else 0
        else:
            va_mae = va_mse = va_r2 = float('nan')

        # ── Close lingering figures before creating new scatter ──
        plt.close('all')

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.scatter(train_y, pred_tr, c='blue', alpha=.7,
                   label='Train', s=80, edgecolors='black', linewidth=.5)
        if has_val and len(val_y) > 0:
            ax.scatter(val_y, pred_va, c='orange', alpha=.8,
                       label='Validation', s=80, edgecolors='black', linewidth=.5)
            allv = np.concatenate([train_y, val_y, pred_tr, pred_va])
        else:
            allv = np.concatenate([train_y, pred_tr])

        lo, hi = allv.min() - .5, allv.max() + .5
        ax.plot([lo, hi], [lo, hi], 'r--', lw=3, label='Perfect')

        ax.set_xlabel('True pKa', fontsize=36, fontweight='bold',
                      fontfamily='Times New Roman')
        ax.set_ylabel('Predicted pKa', fontsize=36, fontweight='bold',
                      fontfamily='Times New Roman')
        ax.set_title(
            f'Performance  (Depth={max_depth},  '
            f'N_train={len(train_y)},  N_val={len(val_y)})',
            fontsize=32, fontweight='bold', fontfamily='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=30, width=2)
        for l in ax.get_xticklabels() + ax.get_yticklabels():
            l.set_fontfamily('Times New Roman')
            l.set_fontweight('bold')
        leg = ax.legend(fontsize=28, frameon=True, fancybox=True, shadow=True)
        for t in leg.get_texts():
            t.set_fontfamily('Times New Roman')
            t.set_fontweight('bold')
        ax.grid(True, alpha=.3, linewidth=1)

        va_mae_s = f'{va_mae:.3f}' if not np.isnan(va_mae) else 'N/A'
        va_mse_s = f'{va_mse:.3f}' if not np.isnan(va_mse) else 'N/A'
        va_r2_s = f'{va_r2:.3f}' if not np.isnan(va_r2) else 'N/A'

        box_txt = (f'Train MAE : {tr_mae:.3f}\n'
                   f'Val   MAE : {va_mae_s}\n'
                   f'Train MSE : {tr_mse:.3f}\n'
                   f'Val   MSE : {va_mse_s}\n'
                   f'Train R²  : {tr_r2:.3f}\n'
                   f'Val   R²  : {va_r2_s}\n\n'
                   f'Depth : {max_depth}')
        ax.text(.05, .95, box_txt, transform=ax.transAxes, fontsize=24,
                fontfamily='Times New Roman', fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=.8))
        plt.tight_layout()
        plt.show()

        self.tree_btn.config(state=tk.NORMAL)
        self.predict_btn.config(state=tk.NORMAL)

        print(f"\n{'='*60}")
        print(f"Model trained  (depth = {max_depth})")
        print(f"Features : {valid_f}")
        print(f"Train {len(train_y)} / Val {len(val_y)}")
        print(f"Train MAE {tr_mae:.4f}  Val MAE {va_mae_s}")
        print(f"Train R²  {tr_r2:.4f}  Val R²  {va_r2_s}")
        if has_val and not np.isnan(va_mae):
            if va_mae > tr_mae * 1.5:
                print("⚠️  Large gap → possible overfitting")
            elif va_mae > tr_mae * 1.2:
                print("💡 Moderate gap → consider reducing depth")
            else:
                print("✅ Good balance")
        print('=' * 60)

    # ══════════════════════════════════════════════════
    #   Chemistry explanations
    # ══════════════════════════════════════════════════
    def _get_chem_explanation(self, feat_name: str) -> str:
        table = {
            "Carbon":
                "Longer carbon chain → stronger electron-donating effect "
                "(EDG) → carboxylate anion less stabilised → pKa ↑ (less acidic). "
                "Alkyl groups push electron density toward –COOH, making proton "
                "release harder.",
            "COOH":
                "Each additional –COOH group provides an extra dissociable proton "
                "and exerts an inductive electron-withdrawing effect on neighbouring "
                "groups → overall pKa ↓ (more acidic).",
            "Halogen":
                "Halogens (F, Cl, Br, I) are electron-withdrawing groups (EWG) via "
                "inductive effect. They stabilise the carboxylate anion (–COO⁻) by "
                "delocalising negative charge → pKa ↓. Strength: F > Cl > Br > I.",
            "F":
                "Fluorine has the highest electronegativity (χ = 3.98) of all elements "
                "→ strongest inductive EWG among halogens. Even a single F adjacent to "
                "–COOH can lower pKa by ~1–2 units compared with the parent acid.",
            "Cl":
                "Chlorine is a moderate EWG via inductive effect (χ = 3.16). "
                "Substitution at the α-carbon (directly next to –COOH) produces the "
                "largest pKa reduction; the effect attenuates rapidly at β, γ positions.",
            "Br":
                "Bromine is a weaker EWG than Cl (larger, more polarisable atom, "
                "lower electronegativity χ = 2.96). Still lowers pKa noticeably "
                "relative to the unsubstituted acid.",
            "OH":
                "Hydroxyl (–OH) has a weak inductive EWG effect through the O atom "
                "(χ = 3.44). Positioned close to –COOH it provides mild additional "
                "stabilisation of the carboxylate anion → small pKa decrease.",
            "NO2":
                "Nitro group (–NO₂) is a powerful EWG by both inductive and resonance "
                "effects. It strongly stabilises –COO⁻ → significant pKa ↓. "
                "Nitro-substituted carboxylic acids are among the strongest in their class.",
            "EWG_Flag":
                "Encodes the type of substituent: EWG (+1) stabilises –COO⁻ → pKa ↓; "
                "EDG (−1) destabilises –COO⁻ → pKa ↑; None (0) leaves pKa unchanged. "
                "This is the primary categorical signal about electronic character.",
            "EWG_Pos":
                "Encodes the distance of the substituent from –COOH: "
                "α = 1 (one bond away), β = 2, γ = 3, None = 0. "
                "The inductive effect decays with each intervening carbon: "
                "α effect >> β effect >> γ effect.",
            "EWG_Rank":
                "Numerical rank of EWG strength: None=0, OH=1, I=2, Br=3, Cl=4, "
                "F=6, NO₂=7. Higher rank → stronger electron withdrawal → greater "
                "stabilisation of carboxylate → lower pKa.",
        }
        return table.get(
            feat_name,
            f"Feature '{feat_name}' modifies the electron density distribution "
            f"near the –COOH group, shifting the acid dissociation equilibrium "
            f"and changing the pKa accordingly.")

    # ══════════════════════════════════════════════════
    #  STEP 5 : Decision Tree & Importance
    #  ── Simplified: auto-popup feature importance,
    #     then directly open Molecule Path Tracer ──
    # ══════════════════════════════════════════════════
    def show_chem_tree(self):
        if self.model is None:
            messagebox.showerror("Error", "Train the model first!")
            return
        # Auto-popup feature importance figure, then open path tracer
        self._show_feature_importance()
        self._show_path_tracer()

    # ══════════════════════════════════════════════════
    #  ② Path Tracer
    # ══════════════════════════════════════════════════
    def _show_path_tracer(self):
        import re
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            from PIL import ImageTk
            _has_rdkit = True
        except ImportError:
            _has_rdkit = False

        win = tk.Toplevel(self.root)
        win.title("🔬 Molecule Path Tracer – Step-by-Step Decision Tree Tracing")
        win.geometry("1520x930")

        vf = self._valid_features
        dt = self.model.tree_

        all_mols: list = []
        for df_src, tag in [
            (self.train_df,  "Train"),
            (self.val_df,    "Validation"),
            (self.test_df,   "Test"),
        ]:
            if df_src is None or len(df_src) == 0:
                continue
            for _, row in df_src.iterrows():
                feats = [row.get(f, np.nan) for f in vf]
                if any(pd.isna(v) for v in feats):
                    continue
                all_mols.append((
                    f"[{tag}]  {row.get(self.name_col, '?')}",
                    feats,
                    row.get("pKa", np.nan),
                    row.get(self.smiles_col, ""),
                ))

        if not all_mols:
            messagebox.showerror(
                "Error",
                "No molecules with complete feature data found.\n"
                "Please fill in feature values in Step 2.",
                parent=win)
            win.destroy()
            return

        mol_names = [m[0] for m in all_mols]

        top_bar = tk.Frame(win)
        top_bar.pack(fill=tk.X, padx=10, pady=6)

        paned = tk.PanedWindow(win, orient=tk.HORIZONTAL, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True)

        left  = tk.Frame(paned)
        right = tk.Frame(paned, bg="#F0F8FF")
        paned.add(left,  minsize=850)
        paned.add(right, minsize=450)

        tk.Label(top_bar, text="Select molecule:",
                 font=("Times New Roman", 12, "bold")).pack(side=tk.LEFT, padx=6)
        mol_var = tk.StringVar(value=mol_names[0])
        mol_cbo = ttk.Combobox(
            top_bar, values=mol_names, textvariable=mol_var,
            state="readonly", width=52,
            font=("Times New Roman", 11))
        mol_cbo.pack(side=tk.LEFT, padx=6)

        fig, ax = plt.subplots(figsize=(11, 7))
        artists = plot_tree(
            self.model, ax=ax, feature_names=vf,
            filled=True, impurity=False, rounded=True,
            fontsize=9, precision=2)
        ax.set_title(
            "← Decision path is highlighted as you step through it",
            fontsize=10, fontstyle="italic",
            fontfamily="Times New Roman", color="#666")
        fig.tight_layout()

        mpl_canvas = FigureCanvasTkAgg(fig, master=left)
        mpl_canvas.draw()
        mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Robust node → artist mapping ──
        node_to_art: dict = {}
        _used_arts: set = set()

        for nid in range(dt.node_count):
            if dt.children_left[nid] == _tree.TREE_LEAF:
                continue
            fname  = vf[int(dt.feature[nid])]
            thr    = float(dt.threshold[nid])
            n_samp = int(dt.n_node_samples[nid])
            feat_pat = re.compile(
                r'(?:^|\n)\s*' + re.escape(fname) + r'\s*<=\s*([\d.]+)')
            for aidx, art in enumerate(artists):
                if aidx in _used_arts:
                    continue
                txt = art.get_text()
                fm = feat_pat.search(txt)
                if not fm:
                    continue
                if abs(float(fm.group(1)) - thr) > 0.015:
                    continue
                sm = re.search(r'samples\s*=\s*(\d+)', txt)
                if not sm or int(sm.group(1)) != n_samp:
                    continue
                node_to_art[nid] = aidx
                _used_arts.add(aidx)
                break

        for nid in range(dt.node_count):
            if nid in node_to_art:
                continue
            n_samp = int(dt.n_node_samples[nid])
            val    = float(dt.value[nid][0][0])
            for aidx, art in enumerate(artists):
                if aidx in _used_arts:
                    continue
                txt = art.get_text()
                if '<=' in txt:
                    continue
                sm = re.search(r'samples\s*=\s*(\d+)', txt)
                if not sm or int(sm.group(1)) != n_samp:
                    continue
                vm = re.search(r'value\s*=\s*([\d.]+)', txt)
                if not vm or abs(float(vm.group(1)) - val) > 0.06:
                    continue
                node_to_art[nid] = aidx
                _used_arts.add(aidx)
                break

        if len(node_to_art) < dt.node_count:
            for nid in range(dt.node_count):
                if nid not in node_to_art and nid < len(artists):
                    node_to_art[nid] = nid

        orig_fc: dict = {}
        for aidx, art in enumerate(artists):
            try:
                bp = art.get_bbox_patch()
                orig_fc[aidx] = bp.get_facecolor() if bp else (1, 1, 1, 1)
            except Exception:
                orig_fc[aidx] = (1, 1, 1, 1)

        def _colour_node(node_id, fc, ec, lw):
            aidx = node_to_art.get(node_id)
            if aidx is None or aidx >= len(artists):
                return
            try:
                bp = artists[aidx].get_bbox_patch()
                if bp:
                    bp.set_facecolor(fc)
                    bp.set_edgecolor(ec)
                    bp.set_linewidth(lw)
            except Exception:
                pass

        def reset_colors():
            for nid, aidx in node_to_art.items():
                if aidx >= len(artists):
                    continue
                try:
                    bp = artists[aidx].get_bbox_patch()
                    if bp:
                        bp.set_facecolor(orig_fc.get(aidx, (1, 1, 1, 1)))
                        bp.set_edgecolor("black")
                        bp.set_linewidth(0.8)
                except Exception:
                    pass

        def paint_path(path, step):
            reset_colors()
            for si, nid in enumerate(path):
                if si > step:
                    break
                if si < step:
                    _colour_node(nid, "#90EE90", "#2E7D32", 2.0)
                else:
                    _colour_node(nid, "#FFD700", "#FF0000", 3.5)
            mpl_canvas.draw()

        # ── Right panel ──
        tk.Label(right, text="🔬  Path Tracer",
                 font=("Times New Roman", 14, "bold"),
                 bg="#F0F8FF", fg="#1A237E").pack(pady=8)

        step_lbl = tk.Label(
            right, text="",
            font=("Times New Roman", 11, "bold"),
            bg="#F0F8FF", fg="#E65100")
        step_lbl.pack(pady=4)

        mol_frame = tk.LabelFrame(
            right, text="  Molecule Structure  ",
            font=("Times New Roman", 11, "bold"),
            bg="#FFFFFF", fg="#1A237E", padx=10, pady=6)
        mol_frame.pack(padx=10, pady=6, fill=tk.X)

        mol_img_label = tk.Label(mol_frame, bg="white", relief=tk.FLAT)
        mol_img_label.pack(pady=4)

        mol_smiles_lbl = tk.Label(
            mol_frame, text="", font=("Courier New", 9),
            bg="#FFFFFF", fg="#555", wraplength=380)
        mol_smiles_lbl.pack(pady=2)

        mol_feats_lbl = tk.Label(
            mol_frame, text="", font=("Courier New", 9),
            bg="#FFFFFF", fg="#333", justify=tk.LEFT, anchor="w")
        mol_feats_lbl.pack(pady=2, fill=tk.X, padx=4)

        txt_frame = tk.Frame(right, bg="#F0F8FF")
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)

        info_txt = tk.Text(
            txt_frame, font=("Times New Roman", 10),
            width=46, height=12, wrap=tk.WORD, bg="#FAFAFA",
            relief=tk.GROOVE, padx=8, pady=6,
            state=tk.DISABLED)
        txt_vsb = ttk.Scrollbar(txt_frame, orient="vertical",
                                 command=info_txt.yview)
        info_txt.configure(yscrollcommand=txt_vsb.set)
        info_txt.grid(row=0, column=0, sticky="nsew")
        txt_vsb.grid(row=0, column=1, sticky="ns")

        info_txt.tag_config("h",    foreground="#1A237E",
                            font=("Times New Roman", 10, "bold"))
        info_txt.tag_config("ok",   foreground="#2E7D32",
                            font=("Times New Roman", 10, "bold"))
        info_txt.tag_config("bad",  foreground="#C62828",
                            font=("Times New Roman", 10, "bold"))
        info_txt.tag_config("chem", foreground="#6A1B9A",
                            font=("Times New Roman", 9, "italic"))
        info_txt.tag_config("sep",  foreground="#AAAAAA")
        info_txt.tag_config("mono", font=("Courier New", 9))

        nav = tk.Frame(right, bg="#F0F8FF")
        nav.pack(pady=10)
        prev_btn = tk.Button(
            nav, text="◀  Previous Step",
            font=("Times New Roman", 10, "bold"),
            bg="#FF9800", fg="white", width=14, pady=4)
        prev_btn.pack(side=tk.LEFT, padx=8)
        next_btn = tk.Button(
            nav, text="Next Step  ▶",
            font=("Times New Roman", 10, "bold"),
            bg="#4CAF50", fg="white", width=14, pady=4)
        next_btn.pack(side=tk.LEFT, padx=8)

        state: dict = {
            "path": [], "step": 0, "feats": [],
            "pka": np.nan, "smiles": "",
        }

        def _write(text, tag=""):
            info_txt.config(state=tk.NORMAL)
            info_txt.insert(tk.END, text, tag)
            info_txt.config(state=tk.DISABLED)

        def _clear():
            info_txt.config(state=tk.NORMAL)
            info_txt.delete(1.0, tk.END)
            info_txt.config(state=tk.DISABLED)

        def _update_mol_image(smiles, feats):
            mol_smiles_lbl.config(
                text=f"SMILES: {smiles}" if smiles else "SMILES: N/A")
            lines = []
            for i, fn in enumerate(vf):
                v = feats[i]
                lines.append(f"  {fn:<12s} = {v}")
            mol_feats_lbl.config(text="\n".join(lines))

            if not _has_rdkit or not smiles:
                mol_img_label.config(
                    image='',
                    text="No 2D structure" if not _has_rdkit else "No SMILES provided",
                    font=("Times New Roman", 10, "italic"), fg="#999")
                return

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    mol_img_label.config(
                        image='', text="Could not parse SMILES",
                        font=("Times New Roman", 10, "italic"), fg="#C62828")
                    return
                img = Draw.MolToImage(mol, size=(360, 220))
                photo = ImageTk.PhotoImage(img)
                mol_img_label.config(image=photo, text="")
                mol_img_label.image = photo
            except Exception as exc:
                mol_img_label.config(
                    image='', text=f"Render error:\n{exc}",
                    font=("Times New Roman", 9), fg="#C62828")

        def show_step():
            path  = state["path"]
            step  = state["step"]
            feats = state["feats"]
            pka   = state["pka"]
            if not path:
                return

            nid     = path[step]
            total   = len(path)
            is_leaf = dt.children_left[nid] == _tree.TREE_LEAF
            n_samp  = int(dt.n_node_samples[nid])
            pred    = float(dt.value[nid][0][0])

            step_lbl.config(
                text=f"Step {step + 1}  of  {total}   |   Node #{nid}")

            _clear()
            mol_display = mol_var.get().split("]  ", 1)[-1]

            _write("━" * 42 + "\n", "sep")
            _write(f"Molecule:  {mol_display}\n", "h")
            if pd.notna(pka):
                _write(f"True pKa:  {float(pka):.2f}\n", "mono")
            _write("━" * 42 + "\n\n", "sep")

            if is_leaf:
                _write("🍃  LEAF NODE – Final Prediction\n\n", "h")
                _write(f"  Predicted pKa  =  {pred:.3f}\n\n", "ok")
                _write(f"  Training samples at this leaf: {n_samp}\n\n")
                _write(
                    "  This node outputs the pKa value above for\n"
                    "  every molecule that arrives here.  No further\n"
                    "  splitting occurs.\n\n", "chem")
                if pd.notna(pka):
                    err = abs(float(pka) - pred)
                    _write(f"  Absolute error  =  {err:.3f}   ", "mono")
                    if err < 0.30:
                        _write("✅  Excellent prediction\n", "ok")
                    elif err < 0.80:
                        _write("🆗  Acceptable prediction\n", "ok")
                    else:
                        _write("⚠   Large error – model uncertain here\n", "bad")
            else:
                fidx      = int(dt.feature[nid])
                thr       = float(dt.threshold[nid])
                fname     = vf[fidx]
                mval      = feats[fidx]
                goes_left = mval <= thr

                _write("🔀  SPLIT NODE – Decision Question\n\n", "h")
                _write(f"  Question:  Is  {fname}  ≤  {thr:.3f} ?\n\n", "h")
                _write(f"  Feature examined  :  {fname}\n")
                _write(f"  This molecule     :  {fname} = {mval}\n", "mono")
                _write(f"  Tree threshold    :  {thr:.3f}\n\n", "mono")

                if goes_left:
                    _write(
                        f"  {mval} ≤ {thr:.3f}  →  Answer: YES\n"
                        f"  → Goes to the LEFT branch  ✅\n\n", "ok")
                else:
                    _write(
                        f"  {mval} > {thr:.3f}  →  Answer: NO\n"
                        f"  → Goes to the RIGHT branch  ❌\n\n", "bad")

                _write(f"  Samples entering this node: {n_samp}\n")
                _write(f"  Average pKa of those samples: {pred:.3f}\n\n")
                _write("🧪  Why does this feature matter?\n\n", "h")
                _write(self._get_chem_explanation(fname) + "\n", "chem")

            prev_btn.config(state=tk.NORMAL if step > 0         else tk.DISABLED)
            next_btn.config(state=tk.NORMAL if step < total - 1 else tk.DISABLED)
            paint_path(path, step)

        def load_mol(event=None):
            sel_name = mol_var.get()
            try:
                idx = mol_names.index(sel_name)
            except ValueError:
                return
            _, feats, pka, smiles = all_mols[idx]
            X_arr = np.array([feats])
            path  = self.model.decision_path(X_arr)[0].indices.tolist()
            state["path"]   = path
            state["step"]   = 0
            state["feats"]  = feats
            state["pka"]    = pka
            state["smiles"] = smiles
            _update_mol_image(smiles, feats)
            show_step()

        def go_prev():
            if state["step"] > 0:
                state["step"] -= 1
                show_step()

        def go_next():
            if state["step"] < len(state["path"]) - 1:
                state["step"] += 1
                show_step()

        prev_btn.config(command=go_prev)
        next_btn.config(command=go_next)
        mol_cbo.bind("<<ComboboxSelected>>", load_mol)
        load_mol()

    # ══════════════════════════════════════════════════
    #  Feature Importance (auto-popup)
    # ══════════════════════════════════════════════════
    def _show_feature_importance(self):
        if self.model is None:
            return
        vf  = self._valid_features
        imp = self.model.feature_importances_

        fig, ax = plt.subplots(figsize=(8, 8))
        idx = np.argsort(imp)[::-1]
        sf  = [vf[i] for i in idx]
        si  = imp[idx]

        palette = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#8B4513',
                   '#9932CC', '#DC143C', '#FF6347', '#32CD32', '#4169E1']
        colors = [palette[i % len(palette)] for i in range(len(sf))]

        bars = ax.bar(range(len(sf)), si, color=colors,
                      edgecolor='black', linewidth=1.5)
        clean_f = [f.split('(')[0].strip() for f in sf]

        ax.set_xlabel('Molecular Descriptors', fontsize=24,
                      fontweight='bold', fontfamily='Times New Roman')
        ax.set_ylabel('Feature Importance', fontsize=24,
                      fontweight='bold', fontfamily='Times New Roman')
        ax.set_title(f'Feature Importance  (Depth = {self.selected_depth})',
                     fontsize=24, fontweight='bold',
                     fontfamily='Times New Roman')
        ax.set_xticks(range(len(sf)))
        ax.set_xticklabels(clean_f, rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', labelsize=24, width=2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily('Times New Roman')
            lbl.set_fontweight('bold')

        for b, v in zip(bars, si):
            if v > 0.001:
                ax.text(b.get_x() + b.get_width() / 2.,
                        b.get_height() + 0.01,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontweight='bold', fontfamily='Times New Roman',
                        fontsize=24)

        ax.grid(axis='y', alpha=.3, linewidth=1)
        ax.set_ylim(0, max(si) * 1.2 if max(si) > 0 else 1)
        plt.tight_layout()
        fn = f"feature_importance_depth_{self.selected_depth}.png"
        plt.savefig(fn, dpi=300, bbox_inches='tight')
        print(f"Saved: {fn}")
        plt.show()

    # ══════════════════════════════════════════════════
    #   STEP 6 : Predict Test Set  (2×2 grid layout)
    # ══════════════════════════════════════════════════
    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Train the model first!")
            return
        if self.test_df is None or len(self.test_df) == 0:
            messagebox.showerror("Error", "No test molecules found!")
            return

        vf = self._valid_features

        pw = tk.Toplevel(self.root)
        pw.title("Step 6 : Test Set Prediction – Manual Feature Entry")
        pw.geometry("1200x760")

        canvas = tk.Canvas(pw)
        sb = ttk.Scrollbar(pw, orient="vertical", command=canvas.yview)
        sf = tk.Frame(canvas)
        sf.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)

        tk.Label(sf, text="Test Set Predictions – Manual Feature Entry",
                 font=("Times New Roman", 16, "bold"), fg="red").pack(pady=8)
        tk.Label(sf,
                 text=("Look at each molecule's structure carefully,\n"
                       "enter the correct feature values, then click "
                       "\"Check & Predict\".\n"
                       "Incorrect values will be highlighted with hints."),
                 font=("Times New Roman", 11, "italic"), fg="#555"
                 ).pack(pady=4)

        mol_entries = {}
        mol_status = {}
        mol_hints_lbl = {}
        pred_labels = {}

        # ── 2×2 grid container ──
        grid_frame = tk.Frame(sf, bg="#FFF5F5")
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)

        for i, (idx, row) in enumerate(self.test_df.iterrows()):
            cell_r, cell_c = divmod(i, 2)
            grid_frame.grid_rowconfigure(cell_r, weight=1)

            pka_s = (f"{row['pKa']:.2f}"
                     if pd.notna(row.get('pKa')) else "?")
            lf = tk.LabelFrame(
                grid_frame,
                text=f"  {row[self.name_col]}   (True pKa : {pka_s})  ",
                font=("Times New Roman", 11, "bold"), fg="red",
                bg="#FFE4E1")
            lf.grid(row=cell_r, column=cell_c,
                    padx=8, pady=8, sticky="nsew")

            inner = tk.Frame(lf, bg="#FFE4E1")
            inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

            # ── Molecule structure image ──
            mol = Chem.MolFromSmiles(row[self.smiles_col])
            if mol:
                img = Draw.MolToImage(mol, size=(155, 155))
                itk = ImageTk.PhotoImage(img)
                il = tk.Label(inner, image=itk, bg="#FFE4E1")
                il.image = itk
                il.pack(side=tk.LEFT, padx=6, anchor="n", pady=4)

            # ── Right side: feature entries + prediction label ──
            right_f = tk.Frame(inner, bg="#FFE4E1")
            right_f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=2)

            ff = tk.Frame(right_f, bg="#FFE4E1")
            ff.pack(fill=tk.X, anchor="n")

            entries = {}
            statuses = {}
            hints = {}

            for feat in vf:
                r_w = tk.Frame(ff, bg="#FFE4E1")
                r_w.pack(fill=tk.X, pady=1)
                tk.Label(r_w, text=feat, width=12, anchor="w",
                         font=("Times New Roman", 10, "bold"),
                         bg="#FFE4E1").pack(side=tk.LEFT)
                ent = tk.Entry(r_w, width=8, font=("Times New Roman", 10))
                ent.pack(side=tk.LEFT, padx=3)

                sl = tk.Label(r_w, text="", width=2,
                              font=("Times New Roman", 11), bg="#FFE4E1")
                sl.pack(side=tk.LEFT, padx=1)
                hl = tk.Label(r_w, text="",
                              font=("Times New Roman", 8, "italic"),
                              bg="#FFE4E1", fg="#888",
                              wraplength=210, anchor="w", justify=tk.LEFT)
                hl.pack(side=tk.LEFT, padx=2)

                entries[feat] = ent
                statuses[feat] = sl
                hints[feat] = hl

            mol_entries[idx] = entries
            mol_status[idx] = statuses
            mol_hints_lbl[idx] = hints

            # Prediction result label
            pl = tk.Label(right_f, text="Prediction : ——",
                          font=("Times New Roman", 12, "bold"),
                          bg="#FFE4E1", fg="blue", wraplength=200)
            pl.pack(pady=6, anchor="w", padx=6)
            pred_labels[idx] = pl

        # ── Check & Predict ──
        def do_check_predict():
            for idx in mol_entries:
                all_correct = True
                vals = []

                for feat in vf:
                    ent = mol_entries[idx][feat]
                    sl  = mol_status[idx][feat]
                    hl  = mol_hints_lbl[idx][feat]
                    s   = ent.get().strip()

                    ent.config(bg="white")
                    sl.config(text="")
                    hl.config(text="")

                    if not s:
                        sl.config(text="❌", fg="red")
                        hl.config(text="Please enter a value", fg="red")
                        ent.config(bg="#FFCCCC")
                        all_correct = False
                        continue

                    try:
                        entered_val = float(s)
                    except ValueError:
                        sl.config(text="❌", fg="red")
                        hl.config(text="Must be a number", fg="red")
                        ent.config(bg="#FFCCCC")
                        all_correct = False
                        continue

                    actual_val = (self.test_df.loc[idx, feat]
                                  if feat in self.test_df.columns
                                  else np.nan)

                    if pd.notna(actual_val):
                        try:
                            actual_float = float(actual_val)
                            if abs(entered_val - actual_float) < 0.01:
                                sl.config(text="✅", fg="green")
                                ent.config(bg="#CCFFCC")
                                hl.config(text="")
                                vals.append(entered_val)
                            else:
                                sl.config(text="❌", fg="red")
                                hint = self.feature_hints.get(
                                    feat, "Check the molecular structure carefully")
                                hl.config(
                                    text=f"Incorrect!  Hint: {hint}",
                                    fg="red")
                                ent.config(bg="#FFCCCC")
                                all_correct = False
                        except (ValueError, TypeError):
                            vals.append(entered_val)
                            sl.config(text="⚠", fg="orange")
                            hl.config(text="Cannot validate", fg="orange")
                    else:
                        vals.append(entered_val)
                        sl.config(text="✅", fg="green")
                        ent.config(bg="#CCFFCC")
                        hl.config(text="")

                if all_correct and len(vals) == len(vf):
                    pr = self.model.predict([vals])[0]
                    true_v = self.test_df.loc[idx, 'pKa']
                    if pd.notna(true_v):
                        err = abs(true_v - pr)
                        clr = ("green" if err < 0.5
                               else ("orange" if err < 1.0 else "red"))
                        pred_labels[idx].config(
                            text=(f"Pred: {pr:.2f}\n"
                                  f"True: {true_v:.2f}\n"
                                  f"Error: {err:.2f}"),
                            fg=clr)
                    else:
                        pred_labels[idx].config(
                            text=f"Pred: {pr:.2f}", fg="blue")
                else:
                    pred_labels[idx].config(
                        text="⚠ Fix incorrect\nvalues first!",
                        fg="red")

        tk.Button(sf, text="🔮  Check & Predict", command=do_check_predict,
                  font=("Times New Roman", 14, "bold"),
                  bg="#2196F3", fg="white",
                  padx=30, pady=8).pack(pady=15)

        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def _mw(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind("<Enter>",
                    lambda e: canvas.bind_all("<MouseWheel>", _mw))
        canvas.bind("<Leave>",
                    lambda e: canvas.unbind_all("<MouseWheel>"))


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()