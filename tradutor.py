import os
import sys
import json
import time
import tempfile
from pathlib import Path
from ebooklib import epub
from bs4 import BeautifulSoup
from transformers import pipeline
from tqdm import tqdm
import re

# Importações essenciais no topo
import torch
from transformers import pipeline

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QProgressBar, QMessageBox, QGroupBox,
                             QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

class TranslationWorker(QThread):
    """Thread para processamento da tradução em segundo plano"""
    progress_updated = pyqtSignal(int, int, str)
    translation_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, input_path, output_path, model_name):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.is_running = True
        
    def run(self):
        try:
            translator = EbookTranslator(self.model_name)
            translator.translate_ebook(self.input_path, self.output_path, self)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        self.is_running = False

class EbookTranslator:
    def __init__(self, model_name=None):
        # Lista de modelos prioritários PARA INGLÊS → PORTUGUÊS
        self.model_options = [
            "Helsinki-NLP/opus-mt-tc-big-en-pt",  # 🥇 MELHOR OPÇÃO - Modelo grande
            "Helsinki-NLP/opus-mt-en-pt",         # 🥈 SEGUNDA OPÇÃO - Modelo base
            "Helsinki-NLP/opus-mt-en-ROMANCE",    # 🥉 TERCEIRA OPÇÃO - Multilíngue (inclui PT)
        ]
        self.model_name = model_name
        self.translator = None
        self.progress_file = "translation_progress.json"
        self.batch_size = 2
        self.max_length = 400
        
    def initialize_translator(self):
        """Inicializa o tradutor para Inglês→Português"""
        try:
            print("🌎 Procurando melhor modelo Inglês→Português...")
            
            models_to_try = [
                "Helsinki-NLP/opus-mt-tc-big-en-pt",  # Primeira escolha
                "Helsinki-NLP/opus-mt-en-pt",         # Segunda escolha  
                "Helsinki-NLP/opus-mt-en-ROMANCE",    # Terceira escolha
            ]
            
            if self.model_name and self.model_name not in models_to_try:
                models_to_try.insert(0, self.model_name)
            
            for model in models_to_try:
                try:
                    print(f"🔄 Tentando: {model}")
                    self.translator = pipeline(
                        "translation", 
                        model=model,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    
                    # Teste de tradução específico
                    test_text = "The book is excellent"
                    result = self.translator(test_text, max_length=50)
                    translation = result[0]['translation_text']
                    
                    print(f"✅ {model} funcionando! '{test_text}' → '{translation}'")
                    
                    # Verifica se está em português
                    if any(pt_char in translation for pt_char in 'ãõáéíóúâêîôû'):
                        print("   ✅ Tradução em português detectada")
                    else:
                        print("   ⚠️  Tradução pode não estar em português")
                    
                    self.model_name = model
                    return
                    
                except Exception as e:
                    print(f"❌ {model} falhou: {str(e)[:80]}...")
                    continue
            
            raise Exception("Nenhum modelo Inglês→Português pôde ser carregado")
            
        except Exception as e:
            print(f"💥 Falha crítica: {e}")
            raise
        
    def translate_text(self, text, model_name):
        """Traduz texto com tratamento específico para cada modelo"""
        try:
            if "ROMANCE" in model_name:
                # Para modelos multilíngues, forçar português
                result = self.translator(text, max_length=50)
                translation = result[0]['translation_text']
                
                # Se a tradução não estiver em português, tentar forçar
                if not any(char in translation for char in 'ãõáéíóúâêîôûàèìòù'):
                    print("⚠️  Tradução não parece estar em português, ajustando...")
                    # Adicionar dica para forçar português
                    result = self.translator(f"{text} [PT]", max_length=50)
                    translation = result[0]['translation_text']
                
                return translation
            else:
                # Para modelos específicos EN-PT
                result = self.translator(text, max_length=50)
                return result[0]['translation_text']
                
        except Exception as e:
            print(f"Erro na tradução de teste: {e}")
            return f"[ERRO: {str(e)[:50]}]"
    
    def extract_text_from_epub(self, epub_path):
        book = epub.read_epub(epub_path)
        chapters_text = []

        for item in book.get_items():
            # Força a verificação do caminho do arquivo
            if item.get_name().startswith("Text/") and item.get_name().endswith(".xhtml"):
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator="\n").strip()
                if text:
                    chapters_text.append(text)
        
        return '\n\n'.join(chapters_text)
    
    def split_into_paragraphs(self, text):
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"translated_paragraphs": [], "current_index": 0, "total_paragraphs": 0}
        return {"translated_paragraphs": [], "current_index": 0, "total_paragraphs": 0}
    
    def save_progress(self, progress_data):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    def translate_batch(self, paragraphs, callback=None):
        if not self.translator:
            self.initialize_translator()
        
        try:
            if "ROMANCE" in self.model_name:
                # Para modelos multilíngues, adicionar dica de idioma
                paragraphs_with_hint = [f"{text} [PT]" for text in paragraphs]
                translations = self.translator(
                    paragraphs_with_hint,
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                    truncation=True
                )
            else:
                # Para modelos específicos EN-PT
                translations = self.translator(
                    paragraphs,
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                    truncation=True
                )
            
            translated_texts = [t['translation_text'] for t in translations]
            return translated_texts
            
        except Exception as e:
            print(f"Erro na tradução: {e}")
            return paragraphs  # Retorna o texto original em caso de erro

    def translate_ebook(self, input_file, output_file, callback=None):
        """Traduz o ebook com callback para atualização de progresso"""
        try:
            # Extrai texto do EPUB
            print("Extraindo texto do EPUB...")
            if callback:
                callback.progress_updated.emit(0, 100, "Extraindo texto do EPUB...")
            
            text_content = self.extract_text_from_epub(input_file)
            paragraphs = self.split_into_paragraphs(text_content)
            total_paragraphs = len(paragraphs)
            
            print(f"Total de parágrafos: {total_paragraphs}")
            
            # Carrega progresso
            progress = self.load_progress()
            translated_paragraphs = progress.get("translated_paragraphs", [])
            start_index = progress.get("current_index", 0)
            
            # Verifica se precisa recomeçar
            if progress.get("total_paragraphs", 0) != total_paragraphs:
                start_index = 0
                translated_paragraphs = []
            
            # Inicializa tradutor
            self.initialize_translator()
            
            if callback:
                callback.progress_updated.emit(start_index, total_paragraphs, "Iniciando tradução...")
            
            # Processa os parágrafos
            for i in range(start_index, total_paragraphs, self.batch_size):
                if callback and not callback.is_running:
                    break
                
                batch = paragraphs[i:min(i + self.batch_size, total_paragraphs)]
                
                # Traduz o lote
                translated_batch = self.translate_batch(batch)
                translated_paragraphs.extend(translated_batch)
                
                # Atualiza progresso
                progress = {
                    "translated_paragraphs": translated_paragraphs,
                    "current_index": i + len(batch),
                    "total_paragraphs": total_paragraphs
                }
                self.save_progress(progress)
                
                # Emite sinal de progresso
                if callback:
                    progress_percent = int((i + len(batch)) / total_paragraphs * 100)
                    callback.progress_updated.emit(
                        i + len(batch), 
                        total_paragraphs, 
                        f"Traduzindo: {i + len(batch)}/{total_paragraphs}"
                    )
                
                time.sleep(0.1)
            
            # Salva resultado final se a tradução foi completada
            if callback and callback.is_running:
                self.save_translated_epub(translated_paragraphs, output_file)
                if os.path.exists(self.progress_file):
                    os.remove(self.progress_file)
                
                if callback:
                    callback.translation_finished.emit(output_file)
            
        except Exception as e:
            if callback:
                callback.error_occurred.emit(str(e))
    
    def save_translated_epub(self, translated_paragraphs, output_file):
        """Cria um EPUB simples com o texto traduzido"""
        book = epub.EpubBook()
        
        # Metadados
        book.set_title('Livro Traduzido')
        book.set_language('pt')
        
        # Cria capítulo com texto traduzido
        translated_text = '\n\n'.join(translated_paragraphs)
        chapter = epub.EpubHtml(
            title='Conteúdo Traduzido',
            file_name='chapter.xhtml',
            lang='pt'
        )
        chapter.content = f'<html><body>{translated_text.replace(chr(10), "<br/>")}</body></html>'
        
        # Adiciona capítulo ao livro
        book.add_item(chapter)
        book.spine = [chapter]
        
        # Salva o EPUB
        epub.write_epub(output_file, book, {})

class TranslationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Tradutor de Ebooks - Inglês para Português')
        self.setGeometry(100, 100, 800, 600)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Grupo de seleção de arquivo
        file_group = QGroupBox("Seleção de Arquivos")
        file_layout = QVBoxLayout()
        
        # Arquivo de entrada
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Ebook de entrada:")
        self.input_path = QLineEdit()
        self.input_path.setReadOnly(True)
        self.input_btn = QPushButton("Selecionar EPUB...")
        self.input_btn.clicked.connect(self.select_input_file)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.input_btn)
        
        # Arquivo de saída
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Salvar como:")
        self.output_path = QLineEdit()
        self.output_btn = QPushButton("Escolher local...")
        self.output_btn.clicked.connect(self.select_output_file)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_btn)
        
        file_layout.addLayout(input_layout)
        file_layout.addLayout(output_layout)
        file_group.setLayout(file_layout)
        
        # Grupo de configurações
        settings_group = QGroupBox("Configurações")
        settings_layout = QHBoxLayout()
        
        self.model_label = QLabel("Modelo de tradução:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Helsinki-NLP/opus-mt-tc-big-en-pt",  # 🥇 MELHOR - Modelo grande
            "Helsinki-NLP/opus-mt-en-pt",         # 🥈 Modelo base
            "Helsinki-NLP/opus-mt-en-ROMANCE",    # 🥉 Multilíngue (fallback)
        ])

        # Tooltips para ajudar o usuário
        self.model_combo.setItemData(0, "Modelo grande - Melhor qualidade", Qt.ToolTipRole)
        self.model_combo.setItemData(1, "Modelo base - Mais rápido", Qt.ToolTipRole)
        self.model_combo.setItemData(2, "Multilíngue - Usa se outros falharem", Qt.ToolTipRole)

        self.model_combo.setCurrentIndex(0)  # Seleciona o melhor modelo por padrão
        
        settings_layout.addWidget(self.model_label)
        settings_layout.addWidget(self.model_combo)
        settings_layout.addStretch()
        settings_group.setLayout(settings_layout)
        
        # Botões de controle
        button_layout = QHBoxLayout()
        self.translate_btn = QPushButton("Iniciar Tradução")
        self.translate_btn.clicked.connect(self.start_translation)
        self.translate_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        
        self.stop_btn = QPushButton("Parar")
        self.stop_btn.clicked.connect(self.stop_translation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        
        self.resume_btn = QPushButton("Retomar")
        self.resume_btn.clicked.connect(self.resume_translation)
        self.resume_btn.setEnabled(os.path.exists("translation_progress.json"))
        
        button_layout.addWidget(self.translate_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.resume_btn)
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Label de status
        self.status_label = QLabel("Pronto para começar")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Área de log
        log_group = QGroupBox("Log de Execução")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        # Adiciona widgets ao layout principal
        layout.addWidget(file_group)
        layout.addWidget(settings_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(log_group)
        
        self.log("Aplicação iniciada. Selecione um arquivo EPUB para começar.")
    
    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar Ebook EPUB", "", "EPUB Files (*.epub)"
        )
        if file_path:
            self.input_path.setText(file_path)
            # Sugere nome de saída automático
            output_path = file_path.replace('.epub', '_traduzido.epub')
            self.output_path.setText(output_path)
            self.log(f"Arquivo selecionado: {file_path}")
    
    def select_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Tradução", "", "EPUB Files (*.epub)"
        )
        if file_path:
            if not file_path.endswith('.epub'):
                file_path += '.epub'
            self.output_path.setText(file_path)
            self.log(f"Arquivo de saída definido: {file_path}")
    
    def start_translation(self):
        input_file = self.input_path.text()
        output_file = self.output_path.text()
        
        if not input_file or not output_file:
            QMessageBox.warning(self, "Aviso", "Selecione os arquivos de entrada e saída!")
            return
        
        if not input_file.endswith('.epub'):
            QMessageBox.warning(self, "Aviso", "O arquivo de entrada deve ser um EPUB!")
            return
        
        self.log("Iniciando tradução...")
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        
        model_name = self.model_combo.currentText()
        self.worker = TranslationWorker(input_file, output_file, model_name)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.translation_finished.connect(self.translation_complete)
        self.worker.error_occurred.connect(self.translation_error)
        self.worker.start()
    
    def stop_translation(self):
        if self.worker:
            self.worker.stop()
            self.log("Tradução interrompida. Você pode retomar depois.")
            self.set_ui_enabled(True)
    
    def resume_translation(self):
        if not os.path.exists("translation_progress.json"):
            QMessageBox.information(self, "Informação", "Nenhum progresso anterior encontrado!")
            return
        
        self.start_translation()
    
    def update_progress(self, current, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)
        self.log(message)
    
    def translation_complete(self, output_file):
        self.log(f"Tradução concluída! Arquivo salvo: {output_file}")
        self.status_label.setText("Tradução concluída com sucesso!")
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(self, "Sucesso", 
                               f"Tradução concluída!\nArquivo salvo em: {output_file}")
    
    def translation_error(self, error_message):
        self.log(f"Erro: {error_message}")
        self.status_label.setText("Erro durante a tradução")
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Erro", f"Ocorreu um erro:\n{error_message}")
    
    def set_ui_enabled(self, enabled):
        self.translate_btn.setEnabled(enabled)
        self.input_btn.setEnabled(enabled)
        self.output_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        self.resume_btn.setEnabled(os.path.exists("translation_progress.json"))
    
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, 'Confirmar saída',
                'A tradução está em andamento. Tem certeza que deseja sair?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    # Verificar dependências essenciais
    try:
        import torch
        from transformers import pipeline
        from PyQt5 import QtWidgets
        print("Todas as dependências estão instaladas! 🎉")
    except ImportError as e:
        print(f"ERRO: Dependência faltando: {e}")
        print("Instale com: pip install torch transformers pyqt5")
        return
    
    app = QApplication(sys.argv)
    
    # Configura estilo visual
    app.setStyle('Fusion')
    
    window = TranslationApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()