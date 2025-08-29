import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup, NavigableString, Comment
from transformers import pipeline
from tqdm import tqdm
import re

# Importações essenciais no topo
import torch

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QProgressBar, QMessageBox, QGroupBox,
                             QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor


# ===========================
# Thread de Tradução
# ===========================
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


# ===========================
# Lógica de Tradução
# ===========================
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

        # parâmetros
        self.batch_size = 8          # mais eficiente para muitos nós curtos
        self.max_length = 400        # limite por item de tradução

        # tags a ignorar (não traduzir)
        self._skip_parent_tags = {"script", "style", "code", "pre", "svg", "math"}
    
    # ---------- Modelo ----------
    def initialize_translator(self):
        """Inicializa o tradutor para Inglês→Português"""
        print("🌎 Procurando melhor modelo Inglês→Português...")
        models_to_try = [
            "Helsinki-NLP/opus-mt-tc-big-en-pt",
            "Helsinki-NLP/opus-mt-en-pt",
            "Helsinki-NLP/opus-mt-en-ROMANCE",
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
                test_text = "The book is excellent"
                result = self.translator(test_text, max_length=50)
                translation = result[0]['translation_text']
                print(f"✅ {model} funcionando! '{test_text}' → '{translation}'")
                if any(pt_char in translation for pt_char in 'ãõáéíóúâêîôûàèìòù'):
                    print("   ✅ Tradução em português detectada")
                else:
                    print("   ⚠️  Tradução pode não estar em português")
                self.model_name = model
                return
            except Exception as e:
                print(f"❌ {model} falhou: {str(e)[:80]}...")
                continue
        raise Exception("Nenhum modelo Inglês→Português pôde ser carregado")

    # ---------- Progresso ----------
    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"doc_index": 0, "node_index": 0, "doc_order": []}
        return {"doc_index": 0, "node_index": 0, "doc_order": []}
    
    def save_progress(self, progress_data):
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    # ---------- Coleta e substituição de nós ----------
    def _gather_text_nodes(self, soup: BeautifulSoup) -> List[NavigableString]:
        """
        Coleta todos os NavigableString traduzíveis, preservando estrutura.
        Ignora comentários, espaços puros e nós dentro de tags a pular.
        """
        text_nodes: List[NavigableString] = []
        for node in soup.find_all(string=True):
            if isinstance(node, Comment):
                continue
            text = str(node)
            if not text or not text.strip():
                continue
            parent = node.parent
            if parent and parent.name and parent.name.lower() in self._skip_parent_tags:
                continue
            # opcional: evitar traduzir atributos ALT de imagens aqui (não estão em NavigableString)
            text_nodes.append(node)
        return text_nodes

    def _batch_translate(self, strings: List[str]) -> List[str]:
        if not strings:
            return []
        if not self.translator:
            self.initialize_translator()
        if "ROMANCE" in self.model_name:
            inputs = [s + " [PT]" for s in strings]
        else:
            inputs = strings
        outputs = self.translator(
            inputs,
            max_length=self.max_length,
            batch_size=self.batch_size,
            truncation=True
        )
        return [o["translation_text"] for o in outputs]

    # ---------- Tradução principal preservando EPUB ----------
    def translate_ebook(self, input_file, output_file, callback: TranslationWorker | None = None):
        """
        Abre o EPUB original, percorre cada DOCUMENT (XHTML), traduz os nós de texto
        e salva um NOVO EPUB preservando:
          - manifest / spine / TOC (índice)
          - diretórios (Images, Styles, Text, etc.)
          - CSS, imagens, fontes, tudo que não for texto
        """
        # 1) Carrega o livro original
        if callback:
            callback.progress_updated.emit(0, 100, "Abrindo EPUB de origem...")
        book = epub.read_epub(input_file)

        # 2) Obtém todos os documentos do spine/manifest (XHTML)
        documents = [it for it in book.get_items_of_type(ITEM_DOCUMENT)]

        # Ordem estável por id/href (útil para retomar)
        documents.sort(key=lambda x: (x.get_id() or "", x.get_name() or ""))

        # 3) Planeja total de nós para barra de progresso
        #    (coletando rapidamente a contagem — pode custar um pouco, mas dá precisão)
        if callback:
            callback.progress_updated.emit(0, 100, "Contando nós de texto...")
        total_nodes = 0
        doc_nodes_cache: Dict[str, int] = {}
        for doc in documents:
            soup = BeautifulSoup(doc.get_content(), 'html.parser')
            n = len(self._gather_text_nodes(soup))
            doc_nodes_cache[doc.get_id()] = n
            total_nodes += n
        if total_nodes == 0:
            # nada para traduzir — apenas copia
            epub.write_epub(output_file, book, {})
            if callback:
                callback.translation_finished.emit(output_file)
            return

        # 4) Carrega/ajusta progresso
        progress = self.load_progress()
        doc_order = [d.get_id() for d in documents]
        if progress.get("doc_order") != doc_order:
            # se a ordem mudou, recomeça
            progress = {"doc_index": 0, "node_index": 0, "doc_order": doc_order}

        current_doc_index = progress["doc_index"]
        current_node_index = progress["node_index"]

        # 5) Inicializa o tradutor
        self.initialize_translator()

        # 6) Loop de tradução por documento / nós
        translated_so_far = sum(doc_nodes_cache[documents[i].get_id()] for i in range(current_doc_index)) + current_node_index
        if callback:
            callback.progress_updated.emit(translated_so_far, total_nodes, "Iniciando tradução...")

        for d_i in range(current_doc_index, len(documents)):
            if callback and not callback.is_running:
                break

            doc = documents[d_i]
            soup = BeautifulSoup(doc.get_content(), 'html.parser')
            nodes = self._gather_text_nodes(soup)

            # retoma do nó onde parou (apenas para o primeiro doc retomado)
            start_node = current_node_index if d_i == current_doc_index else 0

            # processa em lotes
            idx = start_node
            while idx < len(nodes):
                if callback and not callback.is_running:
                    break

                batch_nodes = nodes[idx: idx + self.batch_size]
                batch_texts = [str(n) for n in batch_nodes]

                # traduz
                try:
                    translated = self._batch_translate(batch_texts)
                except Exception as e:
                    # em caso de erro, mantém original para esse batch
                    translated = batch_texts
                    print(f"Erro na tradução do lote: {e}")

                # substitui mantendo estrutura
                for node, new_text in zip(batch_nodes, translated):
                    # preserva espaços à esquerda/direita do nó original
                    left_ws = ""
                    right_ws = ""
                    if str(node).startswith((" ", "\t", "\n")):
                        left_ws = re.match(r"^\s*", str(node)).group(0)
                    if str(node).endswith((" ", "\t", "\n")):
                        right_ws = re.search(r"\s*$", str(node)).group(0)
                    node.replace_with(NavigableString(left_ws + new_text + right_ws))

                # atualiza progresso
                idx += len(batch_nodes)
                translated_so_far += len(batch_nodes)
                current_node_index = idx

                self.save_progress({
                    "doc_index": d_i,
                    "node_index": current_node_index if idx < len(nodes) else 0,
                    "doc_order": doc_order
                })

                if callback:
                    callback.progress_updated.emit(
                        translated_so_far,
                        total_nodes,
                        f"Traduzindo {doc.get_name()}  ({translated_so_far}/{total_nodes})"
                    )

                time.sleep(0.02)

            # fim do documento: grava conteúdo traduzido de volta
            doc.set_content(str(soup).encode("utf-8"))
            current_node_index = 0  # próximo doc começa do zero

        # 7) Se não foi interrompido, salva EPUB preservando estrutura
        if callback and callback.is_running:
            # nada de tocar em toc/spine/manifest — ebooklib reusa tudo do objeto
            # inclusive Imagens, CSS, sources adicionais…
            epub.write_epub(output_file, book, {})

            # limpa progresso
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)

            if callback:
                callback.translation_finished.emit(output_file)


# ===========================
# Interface (PyQt5)
# ===========================
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
        self.model_combo.setItemData(0, "Modelo grande - Melhor qualidade", Qt.ToolTipRole)
        self.model_combo.setItemData(1, "Modelo base - Mais rápido", Qt.ToolTipRole)
        self.model_combo.setItemData(2, "Multilíngue - Usa se outros falharem", Qt.ToolTipRole)
        self.model_combo.setCurrentIndex(0)
        
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


# ===========================
# main()
# ===========================
def main():
    # Verificar dependências essenciais
    try:
        import torch  # noqa: F401
        from transformers import pipeline  # noqa: F401
        from PyQt5 import QtWidgets  # noqa: F401
        print("Todas as dependências estão instaladas! 🎉")
    except ImportError as e:
        print(f"ERRO: Dependência faltando: {e}")
        print("Instale com: pip install torch transformers pyqt5 ebooklib beautifulsoup4")
        return
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = TranslationApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
