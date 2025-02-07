import tensorflow as tf
import matplotlib.pyplot as plt

class PINN:
    def __init__(self, input_shape=2, output_shape=1, layers=[20, 20, 20], activation='tanh', learning_rate=0.01, eq = 'LidDrivenCavity'):
        self.model = self.create_model(input_shape, output_shape, layers, activation)
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule(learning_rate))

        self.eq = eq


    def create_model(self, input_shape,  output_shape, layers, activation):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for units in layers:
            model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dense(output_shape))  # Output layer
        return model

    def learning_rate_schedule(self, initial_learning_rate):
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )

    @tf.function(jit_compile=True)  # This enables XLA compilation
    def train_step(self, loss_function):
        with tf.GradientTape() as tape:
            loss = loss_function()  # Call the loss function here to compute the loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, loss_function, epochs=1000, print_interval=100, autosave_interval=100, plot_loss=False):
        loss_history = []
        epoch_history = []
        
        if plot_loss:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.get_major_formatter().set_scientific(False)
            line, = ax.semilogy([], [], label='Training Loss')
            plt.legend()

        for epoch in range(epochs):
            loss = self.train_step(loss_function)

            if (epoch + 1) % print_interval == 0:
                loss_history.append(loss.numpy())
                epoch_history.append(epoch + 1)

                if plot_loss:
                    line.set_xdata(epoch_history)
                    line.set_ydata(loss_history)
                    ax.relim()  
                    ax.autoscale_view() 
                    plt.draw()
                    plt.pause(0.001)

                print(f"Epoch {epoch + 1}: Loss = {loss.numpy()}")

            if (epoch + 1) % autosave_interval == 0:
                try:
                    self.model.save(f'trainedModels/{self.eq}.tf')
                except OSError as e:
                    print(f"Error saving model: {e}")
                    print("Check disk space and permissions.")
                    raise  # Re-raise the exception so the program doesn't continue with a potentially corrupted model.

        if plot_loss:
            plt.ioff()  # Turn off interactive mode
            plt.close()

    def predict(self, X):
        return self.model.predict(X)
    
    def load(self, model_name):
        import os
        
        filepath = f'trainedModels/{model_name}.tf'

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The specified file does not exist: {filepath}")
        
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model successfully loaded from {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {filepath}: {e}")