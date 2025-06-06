from PySide6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QTabWidget, QTextEdit, QMessageBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pandas as pd
import numpy as np

from preprocessing import inspect_missing_and_summary, normalize_and_encode
from model import (
    split_and_train,
    compute_metrics,
    plot_correlation_heatmap,
    plot_actual_vs_predicted
)


class MplCanvas(FigureCanvas):
    """A Matplotlib canvas embedded in PySide6, with a dark/cyberpunk style."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#1E1E1E")
        self.axes = fig.add_subplot(111, facecolor="#1E1E1E")
        # Darken the axes spines and tick labels
        for spine in self.axes.spines.values():
            spine.set_color("#888888")
        self.axes.tick_params(colors="#CCCCCC", labelsize=10)
        super().__init__(fig)


class MainWindow(QMainWindow):
    """
    Main Window of our app with a Lego–Cyberpunk hybrid theme.
    Contains four tabs:
    1) Upload Data
    2) Data Preprocessing
    3) Model Training
    4) Results
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏗️ House Price Regression (Lego–Cyberpunk) 🏗️")
        self.setGeometry(100, 100, 1000, 750)

        # Store data throughout tabs
        self.df_original = None
        self.df_preprocessed = None
        self.features = None
        self.target = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None

        # Apply the Lego–Cyberpunk style
        self.apply_custom_theme()

        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Initialize each tab
        self.init_upload_tab()
        self.init_preprocessing_tab()
        self.init_model_tab()
        self.init_results_tab()

        # Disable subsequent tabs until prerequisites are met
        self.tabs.setTabEnabled(1, False)  # Preprocessing
        self.tabs.setTabEnabled(2, False)  # Model Training
        self.tabs.setTabEnabled(3, False)  # Results

    def apply_custom_theme(self):
        """
        Apply a custom style sheet blending bright Lego colors with
        neon/cyberpunk accents on a dark background.
        """
        self.setStyleSheet(
            """
            /* Main window: very dark grey (cyberpunk “city” backdrop) */
            QMainWindow {
                background-color: #121212;
            }
            /* Tab bar: neon outlines and larger tabs */
            QTabBar::tab {
                background: #232323;
                color: #00FFA2;
                padding: 12px;
                min-width: 180px;
                font-family: 'Consolas', 'Monospace';
                font-size: 14pt;
                font-weight: bold;
                border: 2px solid #00FFA2;
                margin: 4px;
                border-radius: 6px;
            }
            /* Selected tab: bright neon green */
            QTabBar::tab:selected {
                background: #00FFA2;
                color: #0D0D0D;
            }
            /* Tab pane border */
            QTabWidget::pane {
                border: 4px solid #00FFA2;
                margin-top: 12px;
                background-color: #1E1E1E;
            }
            /* QPushButton: Lego red block with neon glow on hover */
            QPushButton {
                background-color: #D32F2F;
                color: #FFF;
                font-family: 'Consolas', 'Monospace';
                font-size: 12pt;
                font-weight: bold;
                border: 3px solid #B71C1C;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #FF1744;
                border: 3px solid #FF1744;
                box-shadow: 0px 0px 10px #FF1744;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
                border: 3px solid #B71C1C;
            }
            /* QTextEdit & QLabel: pale neon accent on dark background, bold font */
            QTextEdit, QLabel {
                background-color: #1E1E1E;
                color: #E0E0E0;
                font-family: 'Consolas', 'Monospace';
                font-size: 11pt;
                font-weight: bold;
                border: 2px solid #00FFA2;
                border-radius: 6px;
                padding: 8px;
            }
            /* QFileDialog Buttons: inherit QPushButton style */
            """
        )

    # -----------------------------
    # 1) Upload Data Tab
    # -----------------------------
    def init_upload_tab(self):
        """Create the 'Upload Data' tab with a button and preview area."""
        self.upload_tab = QWidget()
        layout = QVBoxLayout()

        lbl = QLabel("🔹 Step 1: Upload your house_prices.csv file 🔹\n"
                     "(Required columns: Size, Location, Number of Rooms, Price)")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        self.upload_button = QPushButton("🔍 Browse CSV")
        self.upload_button.clicked.connect(self.load_csv)
        layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)

        self.upload_preview = QTextEdit()
        self.upload_preview.setReadOnly(True)
        layout.addWidget(self.upload_preview)

        self.upload_tab.setLayout(layout)
        self.tabs.addTab(self.upload_tab, "Upload Data")

    def load_csv(self):
        """
        Open a file dialog. If valid, load CSV into DataFrame,
        show preview, enable Preprocessing tab.
        """
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select house_prices.csv",
            "",
            "CSV Files (*.csv)"
        )
        if not file_name:
            return

        try:
            df = pd.read_csv(file_name)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading CSV", f"Could not load file:\n{e}")
            return

        required_cols = {"Size", "Location", "Number of Rooms", "Price"}
        if not required_cols.issubset(df.columns):
            QMessageBox.critical(
                self,
                "Invalid CSV",
                f"CSV must contain columns: {required_cols}"
            )
            return

        # Save and preview
        self.df_original = df.copy()
        preview_text = df.head(5).to_string(index=False)
        self.upload_preview.setText(f"Data Preview (first 5 rows):\n\n{preview_text}")

        # Enable next tab
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)

    # -----------------------------
    # 2) Data Preprocessing Tab
    # -----------------------------
    def init_preprocessing_tab(self):
        """
        Create the 'Data Preprocessing' tab:
        - Show missing value info & summary statistics
        - Button to run normalization & encoding
        - Preview of preprocessed data
        """
        self.preprocess_tab = QWidget()
        layout = QVBoxLayout()

        lbl = QLabel("🔸 Step 2: Inspect & Preprocess Data 🔸")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        self.missing_summary = QTextEdit()
        self.missing_summary.setReadOnly(True)
        layout.addWidget(self.missing_summary)

        self.run_preprocess_btn = QPushButton("⚙️ Normalize & Encode")
        self.run_preprocess_btn.clicked.connect(self.run_preprocessing)
        layout.addWidget(self.run_preprocess_btn, alignment=Qt.AlignCenter)

        self.preprocessed_preview = QTextEdit()
        self.preprocessed_preview.setReadOnly(True)
        layout.addWidget(self.preprocessed_preview)

        self.preprocess_tab.setLayout(layout)
        self.tabs.addTab(self.preprocess_tab, "Data Preprocessing")

    def run_preprocessing(self):
        """
        - Inspect missing values & summary
        - Normalize 'Size', 'Number of Rooms'; One-hot encode 'Location'
        - Store preprocessed data; show preview; enable Model Training tab
        """
        if self.df_original is None:
            return

        missing_info, summary_stats = inspect_missing_and_summary(self.df_original)
        self.missing_summary.setText(
            f"Missing Values:\n{missing_info}\n\n"
            f"Summary Stats (before preprocess):\n{summary_stats}"
        )

        df_pre, X, y = normalize_and_encode(self.df_original)
        self.df_preprocessed = df_pre.copy()
        self.features = X
        self.target = y

        preview_text = df_pre.head(5).to_string(index=False)
        self.preprocessed_preview.setText(f"Preprocessed Data Preview:\n\n{preview_text}")

        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(2)

    # -----------------------------
    # 3) Model Training Tab
    # -----------------------------
    def init_model_tab(self):
        """
        Create the 'Model Training' tab:
        - Button to train a Linear Regression model
        - Display R² and RMSE
        """
        self.model_tab = QWidget()
        layout = QVBoxLayout()

        lbl = QLabel("🔹 Step 3: Train Linear Regression Model 🔹")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        self.train_model_btn = QPushButton("🚀 Train Model")
        self.train_model_btn.clicked.connect(self.train_model)
        layout.addWidget(self.train_model_btn, alignment=Qt.AlignCenter)

        self.model_metrics = QTextEdit()
        self.model_metrics.setReadOnly(True)
        layout.addWidget(self.model_metrics)

        self.model_tab.setLayout(layout)
        self.tabs.addTab(self.model_tab, "Model Training")

    def train_model(self):
        """
        - Split data (80/20) and train LinearRegression
        - Compute and display R², RMSE
        - Enable Results tab
        """
        if self.features is None or self.target is None:
            return

        model, X_test, y_test, y_pred = split_and_train(self.features, self.target)
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

        r2, rmse = compute_metrics(y_test, y_pred)
        self.model_metrics.setText(
            "Model Performance:\n\n"
            f"• R² (Coefficient of Determination): {r2:.4f}\n"
            f"• RMSE (Error): {rmse:.2f}"
        )

        self.tabs.setTabEnabled(3, True)
        self.tabs.setCurrentIndex(3)

    # -----------------------------
    # 4) Results Tab
    # -----------------------------
    def init_results_tab(self):
        """
        Create the 'Results' tab:
        - Show Correlation Heatmap (with neon accents)
        - Show Actual vs Predicted scatter
        - Each section has a 'Save as JPG' button
        """
        self.results_tab = QWidget()
        main_layout = QVBoxLayout()

        # Correlation Heatmap section
        heatmap_label = QLabel("🔸 Correlation Heatmap 📊")
        heatmap_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(heatmap_label)

        self.heatmap_canvas = MplCanvas(self, width=6, height=5)
        main_layout.addWidget(self.heatmap_canvas)

        save_heatmap_btn = QPushButton("💾 Save Heatmap as JPG")
        save_heatmap_btn.clicked.connect(self.save_heatmap)
        main_layout.addWidget(save_heatmap_btn, alignment=Qt.AlignCenter)

        # Spacer
        main_layout.addSpacing(20)

        # Actual vs Predicted section
        scatter_label = QLabel("🔹 Actual vs Predicted Prices 📈")
        scatter_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(scatter_label)

        self.scatter_canvas = MplCanvas(self, width=6, height=5)
        main_layout.addWidget(self.scatter_canvas)

        save_scatter_btn = QPushButton("💾 Save Scatter Plot as JPG")
        save_scatter_btn.clicked.connect(self.save_scatter)
        main_layout.addWidget(save_scatter_btn, alignment=Qt.AlignCenter)

        self.results_tab.setLayout(main_layout)
        self.tabs.addTab(self.results_tab, "Results")

    def save_heatmap(self):
        """
        Redraw the heatmap (with a cyberpunk colormap),
        prompt for save path, and save as JPG.
        """
        if self.df_preprocessed is None:
            return

        self.heatmap_canvas.axes.clear()
        plot_correlation_heatmap(self.df_preprocessed, self.heatmap_canvas.axes)
        self.heatmap_canvas.draw()

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Heatmap as JPG",
            "",
            "JPG Files (*.jpg);;All Files (*)"
        )
        if path:
            self.heatmap_canvas.figure.savefig(path, format='jpg')
            QMessageBox.information(self, "Saved", f"Heatmap saved to:\n{path}")

    def save_scatter(self):
        """
        Redraw the scatter plot (dark theme, neon accents),
        prompt for save path, and save as JPG.
        """
        if self.y_test is None or self.y_pred is None:
            return

        self.scatter_canvas.axes.clear()
        plot_actual_vs_predicted(self.y_test, self.y_pred, self.scatter_canvas.axes)
        self.scatter_canvas.draw()

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scatter Plot as JPG",
            "",
            "JPG Files (*.jpg);;All Files (*)"
        )
        if path:
            self.scatter_canvas.figure.savefig(path, format='jpg')
            QMessageBox.information(self, "Saved", f"Scatter plot saved to:\n{path}")
