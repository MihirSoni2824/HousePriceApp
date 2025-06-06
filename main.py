import sys
from PySide6.QtWidgets import QApplication
from ui_main import MainWindow

if __name__ == "__main__":
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the main window
    window = MainWindow()
    window.show()

    # Run the Qt event loop
    sys.exit(app.exec())
