"""
GUI for the from-scratch semantic search engine.

Run:  python3 semantic_search_gui.py
"""

import tkinter as tk
from tkinter import ttk

import Semantic_Search as ss


class SemanticSearchGUI:
    WINDOW_TITLE = "Semantic Search Engine (from scratch)"
    WINDOW_SIZE = (1080, 720)

    def __init__(self, root):
        self.root = root
        self.documents = []
        self.embeddings = None
        self.model = None

        root.title(self.WINDOW_TITLE)
        root.geometry(f"{self.WINDOW_SIZE[0]}x{self.WINDOW_SIZE[1]}")
        root.minsize(860, 560)

        self._build_ui()
        self._load_index()
        self._bind_keys()

    # ---------------- UI construction ----------------

    def _build_ui(self):
        # Header
        header = ttk.Frame(self.root, padding=(16, 12, 16, 6))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Semantic Search Engine",
            font=("Helvetica", 18, "bold"),
        ).pack(side=tk.LEFT)
        self.status = ttk.Label(header, text="Not indexed")
        self.status.pack(side=tk.RIGHT)

        # Search row
        search_row = ttk.Frame(self.root, padding=(16, 4, 16, 6))
        search_row.pack(fill=tk.X)
        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(
            search_row, textvariable=self.query_var,
            font=("Helvetica", 12),
        )
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.search_btn = ttk.Button(
            search_row, text="Search", command=self.on_search
        )
        self.search_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Top-k spinbox
        ttk.Label(search_row, text="Top K").pack(side=tk.LEFT, padx=(12, 4))
        self.topk_var = tk.IntVar(value=ss.TOP_K)
        self.topk_spin = ttk.Spinbox(
            search_row, from_=1, to=20, width=4,
            textvariable=self.topk_var,
        )
        self.topk_spin.pack(side=tk.LEFT)

        # Paned: results list (left) + detail (right)
        self.pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.pane.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        left = ttk.Frame(self.pane)
        self.pane.add(left, weight=3)

        cols = ("rank", "score", "file", "category", "title")
        self.tree = ttk.Treeview(left, columns=cols, show="headings",
                                 selectmode="browse")
        heads = {
            "rank": ("#", 40),
            "score": ("Score", 80),
            "file": ("File", 110),
            "category": ("Category", 180),
            "title": ("Title", 360),
        }
        for col, (h, w) in heads.items():
            self.tree.heading(col, text=h)
            self.tree.column(col, width=w, anchor=tk.W if col != "score" else tk.E)
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # Detail pane
        right = ttk.Frame(self.pane)
        self.pane.add(right, weight=2)
        self.detail = tk.Text(right, wrap=tk.WORD, font=("Helvetica", 11),
                              padx=10, pady=10)
        dvsb = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.detail.yview)
        self.detail.configure(yscrollcommand=dvsb.set)
        self.detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dvsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.detail.config(state=tk.DISABLED)

    def _bind_keys(self):
        self.root.bind("<Return>", lambda e: self.on_search())

    # ---------------- indexing ----------------

    def _load_index(self):
        self.status.config(text="Indexing (from scratch)...")
        self.root.update_idletasks()
        try:
            self.documents, self.embeddings, self.model = ss.build_engine()
            ss.validate_documents(self.documents)
        except Exception as exc:  # noqa: BLE001
            self.documents, self.embeddings, self.model = [], None, None
            self.status.config(text=f"Indexing failed: {exc}")
            return
        n = len(self.documents)
        w, g = self.embeddings
        self.status.config(
            text=f"Indexed {n} docs | word-dim {w.shape[1]} | n-gram-dim {g.shape[1]}"
        )
        self.query_entry.focus_set()

    # ---------------- search ----------------

    def on_search(self):
        query = self.query_var.get().strip()
        if not query or not self.embeddings:
            return

        top_k = self.topk_var.get()
        self.status.config(text=f"Searching for «{query}» ...")
        self.root.update_idletasks()

        results = ss.search(query, self.documents, self.embeddings,
                            self.model, top_k=top_k)
        self._last_results = results
        self._populate_tree(results)
        self.status.config(text=f"«{query}» -> {len(results)} results")

    def _populate_tree(self, results):
        self.tree.delete(*self.tree.get_children())
        self.detail.config(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.config(state=tk.DISABLED)

        for rank, r in enumerate(results, start=1):
            fields = r["fields"]
            self.tree.insert(
                "",
                tk.END,
                iid=f"row-{rank}",
                values=(
                    rank,
                    f"{r['score']:.4f}",
                    r["filename"],
                    fields.get("دسته بندی", ""),
                    fields.get("عنوان", ""),
                ),
            )

    def on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        rank = int(self.tree.item(sel[0])["values"][0])
        # map back to full result by running search again is wasteful;
        # instead cache last results.
        if not getattr(self, "_last_results", None):
            return
        if 1 <= rank <= len(self._last_results):
            self._render_detail(self._last_results[rank - 1])

    def _render_detail(self, result):
        fields = result["fields"]
        lines = [
            f"Filename : {result['filename']}",
            f"Score    : {result['score']:.4f}",
            "",
        ]
        for field in ss.METADATA_FIELDS:
            value = fields.get(field, "(ندارد)")
            if value:
                lines.append(f"{field}: {value}")
        sem = result["semantic_text"]
        if sem:
            lines.extend(["", "Semantic content:", sem])

        self.detail.config(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", "\n".join(lines))
        self.detail.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    SemanticSearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()