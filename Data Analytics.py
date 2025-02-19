import os
import sys
import joblib
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QSlider, QProgressBar
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt
from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class DataCleanerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Настройка основного окна
        self.setWindowTitle("Очистка данных, регрессия и генерация отчета")
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("""
            background-color: #c7c5c5;
            border: 2px solid #A0A0A0;
            border-radius: 10px;
        """)  # Легкая тень окна
        
        # Переменные для хранения путей к файлам
        self.file_path = None
        self.save_path = None
        self.dark_mode = False

        # Шрифт
        font = QFont("Roboto Flex", 26)

        # Главный макет
        layout = QVBoxLayout()

        # Темная/светлая тема переключатель
        theme_layout = QHBoxLayout()
        self.theme_slider = QSlider(Qt.Horizontal)
        self.theme_slider.setRange(0, 1)
        self.theme_slider.setValue(0)
        self.theme_slider.valueChanged.connect(self.toggle_theme)
        self.theme_slider.setFixedSize(100, 20)
        theme_layout.addStretch()
        theme_layout.addWidget(self.theme_slider)
        layout.addLayout(theme_layout)

        # Блок "Выбор файла"
        label_file = QLabel("Выберите Excel или CSV файл:")
        label_file.setFont(font)
        layout.addWidget(label_file)
        btn_select_file = QPushButton("ВЫБРАТЬ ФАЙЛ")
        btn_select_file.setFont(font)
        btn_select_file.setStyleSheet("""
            QPushButton {
                background-color: #0F52BA;
                color: white;
                border-radius: 20px;
                padding: 10px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #3C7EDD;
            }
        """)
        btn_select_file.clicked.connect(self.load_file)
        layout.addWidget(btn_select_file)

        # Блок "Выбор папки"
        label_save = QLabel("Выберите место для сохранения отчета и модели:")
        label_save.setFont(font)
        layout.addWidget(label_save)
        btn_select_folder = QPushButton("ВЫБРАТЬ ПАПКУ")
        btn_select_folder.setFont(font)
        btn_select_folder.setStyleSheet("""
            QPushButton {
                background-color: #0F52BA;
                color: white;
                border-radius: 20px;
                padding: 10px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #3C7EDD;
            }
        """)
        btn_select_folder.clicked.connect(self.select_save_location)
        layout.addWidget(btn_select_folder)

        # Блок "Выгрузить отчет"
        btn_export_report = QPushButton("ВЫГРУЗИТЬ ОТЧЕТ")
        btn_export_report.setFont(font)
        btn_export_report.setStyleSheet("""
            QPushButton {
                background-color: #a921ff;
                color: white;
                border-radius: 20px;
                padding: 10px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #b847ff;
            }
        """)
        btn_export_report.clicked.connect(self.process_data)
        layout.addWidget(btn_export_report)

        # Индикатор загрузки
        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(font)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Установка главного макета
        self.setLayout(layout)

    def toggle_theme(self, value):
        if value == 1:
            self.setStyleSheet("""
                background-color: #333333;
                color: white;
                border: 2px solid #505050;
                border-radius: 10px;
            """)
            self.dark_mode = True
        else:
            self.setStyleSheet("""
                background-color: #E0E0E0;
                color: black;
                border: 2px solid #A0A0A0;
                border-radius: 10px;
            """)
            self.dark_mode = False

    def load_file(self):
        """Открывает диалоговое окно для выбора файла."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", "Excel files (*.xlsx *.xls);;CSV files (*.csv)", options=options
        )
        if file_path:
            self.file_path = file_path
            QMessageBox.information(self, "Файл выбран", f"Выбран файл: {file_path}")

    def select_save_location(self):
        """Открывает диалоговое окно для выбора места сохранения."""
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if folder_path:
            self.save_path = folder_path
            QMessageBox.information(self, "Папка выбрана", f"Результаты будут сохранены в: {folder_path}")

    def process_data(self):
        """Основная функция для обработки данных, построения регрессии и генерации отчета."""
        if not self.file_path:
            QMessageBox.critical(self, "Ошибка", "Сначала выберите файл!")
            return
        if not self.save_path:
            QMessageBox.critical(self, "Ошибка", "Сначала выберите место для сохранения!")
            return

        try:
            # Показываем индикатор загрузки
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            progress_step = 10

            # Загрузка данных
            if self.file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(self.file_path)
            elif self.file_path.endswith('.csv'):
                df = pd.read_csv(self.file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла!")

            self.progress_bar.setValue(progress_step)
            progress_step += 10

            # Очистка данных от NaN
            df_cleaned = df.dropna(how='all')

            # Проверка наличия числовых столбцов для регрессии
            numeric_columns = df_cleaned.select_dtypes(include=['number']).columns
            if len(numeric_columns) < 2:
                raise ValueError("Недостаточно числовых столбцов для построения регрессии!")

            # Разделение на признаки (X) и целевую переменную (y)
            X = df_cleaned[numeric_columns[:-1]]
            y = df_cleaned[numeric_columns[-1]]

            self.progress_bar.setValue(progress_step)
            progress_step += 10

            # Модель OLS
            X = sm.add_constant(X)
            model_ols = sm.OLS(y, X).fit()

            # Сохранение модели OLS в PDF
            ols_pdf_name = os.path.basename(self.file_path).split('.')[0] + "_ols_summary.pdf"
            ols_pdf_path = os.path.join(self.save_path, ols_pdf_name)
            with PdfPages(ols_pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(0.01, 0.05, str(model_ols.summary()), fontsize=10, transform=ax.transAxes)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

            self.progress_bar.setValue(progress_step)
            progress_step += 10

            # Генерация отчета
            profile = ProfileReport(df_cleaned, title="Отчет по данным", explorative=True)
            report_name = os.path.basename(self.file_path).split('.')[0] + "_report.html"
            report_path = os.path.join(self.save_path, report_name)
            profile.to_file(report_path)

            self.progress_bar.setValue(progress_step)
            progress_step += 10

            # Генерация графика регрессии и сохранение в PDF
            pdf_name = os.path.basename(self.file_path).split('.')[0] + "_regression.pdf"
            pdf_path = os.path.join(self.save_path, pdf_name)
            with PdfPages(pdf_path) as pdf:
                for i, feature in enumerate(X.columns[1:]):  # Пропускаем константу
                    plt.figure(figsize=(8, 6))
                    plt.scatter(X[feature], y, color='blue', label='Данные')
                    plt.plot(X[feature], model_ols.predict(X), color='red', label='Регрессия')
                    plt.xlabel(feature)
                    plt.ylabel(numeric_columns[-1])
                    plt.title(f"Регрессия для {feature}")
                    plt.legend()
                    pdf.savefig()  # Сохранение страницы в PDF
                    plt.close()

                    # Обновляем прогресс
                    self.progress_bar.setValue(progress_step + int(i / len(X.columns[1:]) * 30))

            # Открытие папки сохранения
            os.startfile(self.save_path)

            # Скрываем индикатор загрузки
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)

            QMessageBox.information(
                self, "Успех",
                f"Отчет успешно создан и сохранен в:\n{report_path}\n\n"
                f"Сводка модели OLS сохранена в:\n{ols_pdf_path}\n\n"
                f"Графики регрессии сохранены в:\n{pdf_path}"
            )

        except Exception as e:
            # Скрываем индикатор загрузки при ошибке
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataCleanerApp()
    window.show()
    sys.exit(app.exec_())