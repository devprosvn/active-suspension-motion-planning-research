import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE

class SuspensionTFTModel:
    def __init__(self, hidden_size=32, lstm_layers=1, dropout=0.1):
        """
        Initialize TFT model for suspension impact prediction
        Args:
            hidden_size: Size of hidden layers
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
    def create_dataset(self, data, time_idx='time_idx', target='body_roll'):
        """
        Create TimeSeriesDataset for TFT model
        Args:
            data: DataFrame containing time series data
            time_idx: Column name for time index
            target: Target variable name
        Returns:
            TimeSeriesDataSet object
        """
        # Define dataset parameters
        max_prediction_length = 6  # Predict next 6 steps
        max_encoder_length = 24    # Use last 24 steps as history
        
        # Define known and observed variables
        known_reals = ['velocity_x', 'velocity_y', 'velocity_z', 
                      'acceleration_x', 'acceleration_y', 'acceleration_z',
                      'steering_angle', 'throttle', 'brake']
        
        observed_reals = ['body_roll']
        
        # Create dataset
        training = TimeSeriesDataSet(
            data,
            time_idx=time_idx,
            target=target,
            group_ids=["terrain_type"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["terrain_type"],
            time_varying_known_reals=known_reals,
            time_varying_observed_reals=observed_reals,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        
        return training
    
    def build_model(self, training):
        """
        Build TFT model with given dataset parameters
        Args:
            training: TimeSeriesDataSet object
        Returns:
            Configured TFT model
        """
        # Create TFT model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout_rate=self.dropout,
            output_size=1,
            loss=MAE(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        return tft
    
    def train_model(self, model, train_dataloader, val_dataloader, epochs=20, lr=0.03):
        """
        Train TFT model
        Args:
            model: TFT model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
        Returns:
            Trained model
        """
        # Configure optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Configure learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, verbose=True
        )
        
        # Train model
        model.fit(
            train_dataloader,
            val_dataloader=val_dataloader,
            max_epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        return model