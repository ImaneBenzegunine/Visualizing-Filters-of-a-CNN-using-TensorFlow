# Visualizing Filters of a Convolutional Neural Network (VGG16)

This repository contains code to visualize the filters of a pre-trained VGG16 model, trained on the ImageNet dataset, using TensorFlow. The code explores how different filters in the convolutional layers respond to random images during a training loop.

## Overview

This project demonstrates how to visualize and understand the different filters of a Convolutional Neural Network (CNN), specifically the VGG16 model. We use the following steps:

1. **Loading Pre-trained VGG16 Model**: The VGG16 model is loaded with pre-trained weights from the ImageNet dataset, and the top layer is removed to only use the convolutional layers.
   
2. **Get Layer Outputs**: A submodel is created that outputs the activations of any convolutional layer. This allows us to visualize how different filters behave during training.

3. **Image Visualization**: A random image is created and normalized, providing an initial input to the network.

4. **Training Loop for Visualization**: A training loop is implemented to optimize the random image to maximize the activation of a specific filter in a chosen layer. This helps in visualizing what kind of patterns or features are activated by each filter.

5. **Final Results**: The results are plotted and can help to understand the learned filters of the network.

## Requirements

To run the code, you'll need the following dependencies installed:

- Python 3.x
- TensorFlow
- Matplotlib

You can install the required dependencies using pip:

```bash
pip install tensorflow matplotlib
```

## Running the Code

### Step 1: Install Dependencies

Make sure you have TensorFlow and Matplotlib installed. You can install them using the following command:

```bash
pip install tensorflow matplotlib
```

### Step 2: Load the Pre-trained VGG16 Model

The model is loaded using TensorFlow's Keras API with the following code:

```python
import tensorflow as tf

model = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet',
    input_shape=(96, 96, 3)
)
model.summary()
```

This will load the VGG16 model without the top layer (`include_top=False`), using the ImageNet weights.

### Step 3: Get Submodel Output

The `get_submodel` function is used to extract outputs from the desired convolutional layer. This allows us to visualize the feature maps from any of the convolutional layers.

```python
def get_submodel(layer_name):
    return tf.keras.models.Model(
        model.input,
        model.get_layer(layer_name).output
    )
```

### Step 4: Visualize Filters

The `visualize_filter` function allows you to visualize how a particular filter (from a specified layer) behaves by optimizing a random image to maximize the filter's activation. You can specify the layer and filter index, and the code will run a training loop to modify the input image.

```python
def visualize_filter(layer_name, f_index=None, iters=50):
    submodel = get_submodel(layer_name)
    num_filters = submodel.output.shape[-1]

    if f_index is None:
        f_index = random.randint(0, num_filters-1)
    assert num_filters > f_index, 'f_index is out of bounds'

    image = create_image()
    verbose_step = int(iters / 10)

    for i in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(image)
            out = submodel(tf.expand_dims(image, axis=0))[:, :, :, f_index]
            loss = tf.math.reduce_mean(out)
        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image += grads * 10

        if (i + 1) % verbose_step == 0:
            print(f'Iteration: {i + 1}, Loss: {loss.numpy():.4f}')
    plot_image(image, f'{layer_name}, {f_index}')
```

### Step 5: Visualize Specific Layer Filters

You can visualize filters from different layers. The following layers are available:

- block1_conv1
- block1_conv2
- block2_conv1
- block2_conv2
- block3_conv1
- block3_conv2
- block3_conv3
- block4_conv1
- block4_conv2
- block4_conv3
- block5_conv1
- block5_conv2
- block5_conv3

Example usage:

```python
layer_name = 'block3_conv1'
visualize_filter(layer_name, iters=100)
```

This will visualize the filters from the `block3_conv1` layer after 100 iterations.

## Example Output

The output consists of visualizations of the filters at different layers. Each filter is shown as an image, which represents the patterns that the filter has learned to recognize.
