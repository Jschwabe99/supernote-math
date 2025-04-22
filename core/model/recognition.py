"""CNN + RNN/Transformer model for handwritten math recognition."""

import tensorflow as tf
from tensorflow.keras import layers, Model
import platform
import os

from config import ModelConfig


class MathRecognitionModel:
    """CNN + Transformer or CNN + RNN model for handwritten math recognition."""
    
    def __init__(self, input_shape=(128, 128, 1), vocab_size=128, model_type='rnn', config=None):
        """Initialize model architecture.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            vocab_size: Size of vocabulary for output tokens
            model_type: 'rnn' or 'transformer'
            config: Optional model configuration
        """
        self.input_shape = input_shape
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.config = config or ModelConfig()
        
        # Check for Metal and optimize accordingly
        self.is_apple_silicon = platform.system() == "Darwin" and platform.processor() == "arm"
        self.has_metal = len(tf.config.list_physical_devices('GPU')) > 0 and self.is_apple_silicon
        
        if self.has_metal:
            print("Building model optimized for Apple Metal GPU")
    
    def _build_cnn_encoder(self, inputs):
        """Build CNN encoder for feature extraction.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Encoded features tensor
        """
        # Start with sequential CNN blocks
        x = inputs
        
        # Configure CNN parameters based on model config
        filters = self.config.cnn_filters
        kernel_size = self.config.cnn_kernel_size
        pool_size = self.config.cnn_pool_size
        
        # CNN Blocks
        for filter_size in filters:
            x = layers.Conv2D(
                filter_size, 
                kernel_size, 
                activation='relu', 
                padding='same'
            )(x)
            
            # Batch normalization helps training stability, especially on GPU
            x = layers.BatchNormalization()(x)
            
            # Max pooling to reduce dimensions
            x = layers.MaxPooling2D(pool_size)(x)
            
            # Add dropout for regularization
            # Use higher dropout rate on Metal GPUs which tend to overfit faster
            dropout_rate = 0.3 if self.has_metal else 0.2
            x = layers.Dropout(dropout_rate)(x)
        
        # Reshape for sequence modeling
        batch_size = tf.shape(inputs)[0]
        feature_dim = x.shape[3]
        sequence_length = x.shape[1] * x.shape[2]
        x = tf.reshape(x, [batch_size, sequence_length, feature_dim])
        
        return x
    
    def build_transformer_model(self):
        """Build model with CNN encoder and Transformer decoder.
        
        Returns:
            TensorFlow Keras Model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN Encoder
        x = self._build_cnn_encoder(inputs)
        
        # Transformer Encoder - use parameters from config
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-head attention with dynamic parameters based on device
        key_dim = self.config.transformer_d_model // self.config.transformer_heads
        x = layers.MultiHeadAttention(
            key_dim=key_dim, 
            num_heads=self.config.transformer_heads,
            dropout=self.config.transformer_dropout
        )(x, x, x, attention_mask=None, training=True)
        
        x = layers.Dropout(self.config.transformer_dropout)(x)
        x = layers.Dense(self.config.transformer_dim_feedforward, activation='relu')(x)
        x = layers.Dense(self.config.transformer_d_model)(x)
        
        # Output Layer
        x = layers.Dense(self.vocab_size, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model
    
    def build_rnn_model(self):
        """Build model with CNN encoder and RNN (LSTM/GRU) decoder.
        
        Returns:
            TensorFlow Keras Model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN Encoder
        x = self._build_cnn_encoder(inputs)
        
        # Embedding layer before RNN
        x = layers.Dense(self.config.embedding_dim)(x)
        
        # RNN Decoder
        # Metal performs better with standard LSTM vs CuDNNLSTM
        # Apple Silicon has unified memory so bidirectional layers perform well
        for i in range(self.config.rnn_layers):
            # For Metal GPUs, use standard bidirectional LSTM
            if self.has_metal:
                return_sequences = True if i < self.config.rnn_layers - 1 else True
                # Use standard LSTM for Metal
                x = layers.Bidirectional(
                    layers.LSTM(
                        self.config.rnn_hidden_size, 
                        return_sequences=return_sequences,
                        dropout=self.config.rnn_dropout,
                        recurrent_dropout=self.config.rnn_dropout/2
                    )
                )(x)
            else:
                # For other GPUs or CPU
                return_sequences = True if i < self.config.rnn_layers - 1 else True
                x = layers.Bidirectional(
                    layers.LSTM(
                        self.config.rnn_hidden_size, 
                        return_sequences=return_sequences
                    )
                )(x)
                x = layers.Dropout(self.config.rnn_dropout)(x)
        
        # Attention mechanism
        # Simple attention mechanism that works well on both CPU and GPU
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(x.shape[-1])(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
        
        # Output Layer
        x = layers.Dense(self.vocab_size, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model
    
    def build_model(self):
        """Build model based on selected architecture type.
        
        Returns:
            TensorFlow Keras Model
        """
        if self.model_type == 'transformer':
            return self.build_transformer_model()
        elif self.model_type == 'rnn':
            return self.build_rnn_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def compile_model(self, model, learning_rate=None):
        """Compile model with appropriate optimizer and loss.
        
        Args:
            model: TensorFlow Keras Model
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled model
        """
        if learning_rate is None:
            learning_rate = 1e-3  # Default learning rate
            
        # Use appropriate optimizer based on device
        if self.has_metal:
            # Adam works well on Metal
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            # RMSprop sometimes performs better on CPU
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


def get_model_summary(model):
    """Get a string representation of model summary.
    
    Args:
        model: TensorFlow Keras Model
        
    Returns:
        String with model summary
    """
    # Redirect model.summary() output to a string
    import io
    summary_string = io.StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    
    return summary_string.getvalue()