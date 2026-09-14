import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def create_multi_task_model(n_features, targets):
    input_layer = Input(shape=(n_features,), name='input_features')
    x = BatchNormalization()(input_layer)
    x = Dropout(0.2)(x)

    # Shared Body
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    shared_body = Dense(64, activation='relu')(x)

    output_layers = []
    output_losses = {}
    output_metrics = {}

    # Create heads
    for target_name in targets:
        head = Dense(16, activation='relu')(shared_body)
        head = Dropout(0.2)(head)
        output = Dense(1, activation='sigmoid', name=target_name)(head)
        
        output_layers.append(output)
        output_losses[target_name] = 'binary_crossentropy'
        output_metrics[target_name] = tf.keras.metrics.AUC(name=f'{target_name}_auc')

    model = Model(inputs=input_layer, outputs=output_layers)
    model.compile(optimizer='adam', loss=output_losses, metrics=output_metrics)
    return model