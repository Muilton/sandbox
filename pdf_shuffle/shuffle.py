#!/usr/bin/python3
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QLineEdit, QMessageBox
import sys
import PyPDF2


class PdfShuffleDesing(QWidget):
    def __init__(self):

        super().__init__()
        self.resize(470, 200)
        self.setWindowTitle('PDF Shuffle for Xerox B610')

        btn_choice = QPushButton('Выберите файл (*.pdf)', self)
        btn_choice.resize(450, 50)
        btn_choice.move(10, 10)
        btn_choice.clicked.connect(self.showDialog)

        self.file_label = QLabel(self)
        self.file_label.move(10, 70)
        self.file_label.resize(550, 20)
        self.file_label.setText('Имя файла')

        btn_shuffle = QPushButton('Отсортировать', self)
        btn_shuffle.resize(450, 50)
        btn_shuffle.move(10, 130)
        btn_shuffle.clicked.connect(self.shuffle)

    def showDialog(self):
        self.file_dialog = QFileDialog.getOpenFileName(self, 'Выберите файл', 'S:\General\ТТН по отгрузкам', '*.pdf')[0]
        if self.file_dialog:
            self.file_label.setText(self.file_dialog)

    def shuffle(self):
        try:
            input_pdf = PyPDF2.PdfFileReader(self.file_dialog)
            output_pdf = PyPDF2.PdfFileWriter()
            filename = self.file_dialog[:-4] + '_shuffled.pdf'

            for i in range(0, input_pdf.getNumPages() - 1, 2):
                output_pdf.addPage(input_pdf.getPage(i + 1))
                output_pdf.addPage(input_pdf.getPage(i))

            with open(filename, 'wb') as out:
                output_pdf.write(out)

            out.close()

        except Exception:
            msg_box = QMessageBox.about(self, "Ошибка", "Файл не выбран!")

        else:
            msg_box = QMessageBox.about(self, "Результат", "Файл отсортирован")

            sys.exit()
           # os.system("start " + filename)


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    app = QApplication(sys.argv)  # Новый экземпляр QApplication
    window = PdfShuffleDesing()  # Создаём объект класса PdfShuffleDesing
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение