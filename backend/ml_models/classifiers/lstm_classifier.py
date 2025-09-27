# backend/ml_models/classifiers/lstm_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .base_classifier import BaseClassifier
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM outputs.
    """

    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to LSTM outputs.

        Args:
            lstm_outputs: LSTM outputs of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_outputs)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

        # Apply attention weights to get context vector
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)  # (batch_size, hidden_size)

        return context_vector, attention_weights.squeeze(-1)


class LSTMAttentionNetwork(nn.Module):
    """
    LSTM with Attention network for text classification.

    This implements the architecture described in the project documentation
    with embedding, bidirectional LSTM, attention, and classification layers.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize the LSTM Attention Network.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMAttentionNetwork, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = AttentionLayer(lstm_output_size)

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask (not used in this implementation)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size * directions)

        # Attention
        context_vector, attention_weights = self.attention(lstm_out)

        # Classification
        output = self.dropout(context_vector)
        logits = self.classifier(output)

        return logits


class LSTMClassifier(BaseClassifier):
    """
    Wrapper for the LSTMAttentionNetwork to provide a standard interface.

    This class manages the LSTM with attention model, including training,
    prediction, and model persistence functionality.
    """

    def __init__(self,
                 model_name: str = "lstm_attention",
                 vocab_size: int = 10000,
                 embedding_dim: int = 300,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 max_length: int = 512,
                 learning_rate: float = 0.001,
                 **kwargs):
        """
        Initialize the LSTM classifier.

        Args:
            model_name: Name identifier for the model
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            max_length: Maximum sequence length
            learning_rate: Learning rate for training
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.max_length = max_length
        self.learning_rate = learning_rate

        # Initialize the network
        self._model = LSTMAttentionNetwork(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )

        # Training components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(self.device)

        # Vocabulary and tokenization
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        logger.info(f"LSTM classifier initialized on device: {self.device}")

    def _build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.

        Args:
            texts: List of training texts
        """
        logger.info("Building vocabulary from training texts")

        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort words by frequency and take top vocab_size - 2 (excluding PAD and UNK)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.vocab_size - 2]

        # Build vocabulary mappings
        for idx, (word, _) in enumerate(top_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        logger.info(f"Vocabulary built with {len(self.word_to_idx)} words")

    def _text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        words = text.lower().split()
        sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is UNK token

        # Truncate or pad to max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence.extend([0] * (self.max_length - len(sequence)))  # 0 is PAD token

        return sequence

    def _prepare_data(self, texts: List[str], labels: Optional[List[int]] = None) -> DataLoader:
        """
        Prepare data for training or inference.

        Args:
            texts: List of text samples
            labels: Optional list of labels

        Returns:
            DataLoader with prepared data
        """
        # Convert texts to sequences
        sequences = [self._text_to_sequence(text) for text in texts]
        input_ids = torch.tensor(sequences, dtype=torch.long)

        if labels is not None:
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(input_ids, labels_tensor)
        else:
            dataset = TensorDataset(input_ids)

        return DataLoader(dataset, batch_size=32, shuffle=(labels is not None))

    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the LSTM classifier with custom PyTorch training loop.

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Optional validation texts
            y_val: Optional validation labels

        Returns:
            Dictionary containing training metrics and history
        """
        logger.info(f"Training LSTM classifier on {len(X_train)} samples")

        # Extract texts from input
        if isinstance(X_train, pd.DataFrame):
            if 'text' in X_train.columns:
                train_texts = X_train['text'].tolist()
            else:
                train_texts = X_train.iloc[:, 0].tolist()
        else:
            train_texts = X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train)

        train_labels = y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train)

        # Build vocabulary
        self._build_vocabulary(train_texts)

        # Prepare data loaders
        train_loader = self._prepare_data(train_texts, train_labels)

        val_loader = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                if 'text' in X_val.columns:
                    val_texts = X_val['text'].tolist()
                else:
                    val_texts = X_val.iloc[:, 0].tolist()
            else:
                val_texts = X_val.tolist() if hasattr(X_val, 'tolist') else list(X_val)

            val_labels = y_val.tolist() if hasattr(y_val, 'tolist') else list(y_val)
            val_loader = self._prepare_data(val_texts, val_labels)

        # Training parameters
        num_epochs = 10
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # Training loop
        self._model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for batch in train_loader:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self._model(input_ids)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            # Calculate epoch metrics
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct_predictions / total_predictions

            training_history['train_loss'].append(epoch_loss)
            training_history['train_acc'].append(epoch_acc)

            # Validation
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, "
                           f"Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        self.is_trained = True

        # Prepare return metrics
        metrics = {
            "training_history": training_history,
            "final_train_loss": training_history['train_loss'][-1],
            "final_train_accuracy": training_history['train_acc'][-1],
            "epochs_completed": num_epochs,
            "vocabulary_size": len(self.word_to_idx)
        }

        if val_loader:
            metrics.update({
                "final_val_loss": training_history['val_loss'][-1],
                "final_val_accuracy": training_history['val_acc'][-1]
            })

        logger.info(f"LSTM training completed. Final training accuracy: {metrics['final_train_accuracy']:.4f}")
        return metrics

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model on validation data.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self._model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)

                outputs = self._model(input_ids)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        self._model.train()
        return total_loss / len(val_loader), correct_predictions / total_predictions

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict class labels for input texts.

        Args:
            texts: List of text samples to classify

        Returns:
            Array of predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self._model.eval()
        data_loader = self._prepare_data(texts)
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch[0].to(self.device)
                outputs = self._model(input_ids)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for input texts.

        Args:
            texts: List of text samples to classify

        Returns:
            Array of prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self._model.eval()
        data_loader = self._prepare_data(texts)
        probabilities = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch[0].to(self.device)
                outputs = self._model(input_ids)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test texts
            y_test: Test labels

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Add AUC-ROC for binary classification
        if self.num_classes == 2:
            from sklearn.metrics import roc_auc_score
            metrics["auc_roc"] = roc_auc_score(y_test, y_proba[:, 1])

        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return metrics

    def save(self, file_path: str) -> None:
        """
        Save the trained model to a file.

        Args:
            file_path: Path where the model should be saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model_state_dict": self._model.state_dict(),
            "model_name": self.model_name,
            "config": self.config,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "is_trained": self.is_trained
        }

        torch.save(model_data, file_path)
        logger.info(f"LSTM model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'LSTMClassifier':
        """
        Load a trained model from a file.

        Args:
            file_path: Path to the saved model

        Returns:
            Loaded classifier instance
        """
        model_data = torch.load(file_path, map_location='cpu')

        # Create new instance
        classifier = cls(
            model_name=model_data["model_name"],
            vocab_size=model_data["vocab_size"],
            embedding_dim=model_data["embedding_dim"],
            hidden_size=model_data["hidden_size"],
            num_layers=model_data["num_layers"],
            num_classes=model_data["num_classes"],
            dropout=model_data["dropout"],
            max_length=model_data["max_length"],
            learning_rate=model_data["learning_rate"],
            **model_data.get("config", {})
        )

        # Restore model state
        classifier._model.load_state_dict(model_data["model_state_dict"])
        classifier.word_to_idx = model_data["word_to_idx"]
        classifier.idx_to_word = model_data["idx_to_word"]
        classifier.is_trained = model_data["is_trained"]

        # Move to appropriate device
        classifier._model.to(classifier.device)

        logger.info(f"LSTM model loaded from {file_path}")
        return classifier