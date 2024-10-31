from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models import HoverTool

# Data for Non-CUDA Transformer (5 iterations)
epochs_5_no_cuda = [1, 2, 3, 4, 5]
loss_5_no_cuda = [0.4985, 0.3452, 0.2903, 0.2488, 0.2169]
accuracy_5_no_cuda = [82.07, 87.98, 90.20, 91.63, 92.65]

# Data for CUDA Transformer (5 iterations, version 1)
epochs_5_cuda_v1 = [1, 2, 3, 4, 5]
loss_5_cuda_v1 = [0.5016, 0.3517, 0.2811, 0.2353, 0.2041]
accuracy_5_cuda_v1 = [82.02, 87.65, 90.32, 92.08, 93.37]

# Data for CUDA Transformer (10 iterations, version 1)
epochs_10_cuda_v1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss_10_cuda_v1 = [0.5105, 0.3642, 0.2884, 0.2441, 0.2137, 0.1940, 0.1776, 0.1637, 0.1537, 0.1453]
accuracy_10_cuda_v1 = [81.80, 87.22, 89.98, 91.59, 92.79, 93.45, 94.21, 94.55, 94.84, 95.17]

# Data for CUDA Transformer (10 iterations, version 2)
epochs_10_cuda_v2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss_10_cuda_v2 = [0.5359, 0.4339, 0.4165, 0.3934, 0.4311, 0.4447, 0.4152, 0.4287, 0.4443, 0.4361]
accuracy_10_cuda_v2 = [80.57, 84.43, 84.85, 85.94, 83.30, 82.89, 84.28, 83.92, 83.70, 84.12]

# Data for CUDA Transformer (200 iterations - subset of first 128 epochs for visualization)
epochs_128_cuda = list(range(1, 129))
loss_128_cuda = [
    0.5158, 0.3532, 0.2829, 0.2364, 0.2061, 0.1816, 0.1661, 0.1491, 0.1352, 0.1259, 
    0.1162, 0.1094, 0.1010, 0.0946, 0.0894, 0.0855, 0.0781, 0.0779, 0.0708, 0.0695, 
    0.0654, 0.0614, 0.0593, 0.0562, 0.0549, 0.0537, 0.0490, 0.0486, 0.0467, 0.0464, 
    0.0439, 0.0426, 0.0406, 0.0411, 0.0379, 0.0374, 0.0380, 0.0363, 0.0343, 0.0353, 
    0.0323, 0.0321, 0.0324, 0.0316, 0.0296, 0.0302, 0.0283, 0.0291, 0.0269, 0.0263, 
    0.0260, 0.0263, 0.0259, 0.0251, 0.0249, 0.0238, 0.0229, 0.0236, 0.0221, 0.0228, 
    0.0228, 0.0210, 0.0219, 0.0203, 0.0210, 0.0211, 0.0196, 0.0212, 0.0206, 0.0205, 
    0.0193, 0.0192, 0.0208, 0.0171, 0.0182, 0.0188, 0.0184, 0.0182, 0.0175, 0.0183, 
    0.0177, 0.0169, 0.0178, 0.0173, 0.0164, 0.0155, 0.0156, 0.0160, 0.0157, 0.0158, 
    0.0151, 0.0149, 0.0148, 0.0152, 0.0152, 0.0135, 0.0151, 0.0154, 0.0146, 0.0138, 
    0.0137, 0.0139, 0.0151, 0.0139, 0.0134, 0.0137, 0.0137, 0.0134, 0.0143, 0.0130, 
    0.0128, 0.0116, 0.0126, 0.0130, 0.0131, 0.0120, 0.0120, 0.0123, 0.0115, 0.0110, 
    0.0130, 0.0109, 0.0127, 0.0113, 0.0117, 0.0119, 0.0106, 0.0117
]
accuracy_128_cuda = [
    81.79, 87.61, 90.12, 91.83, 93.05, 93.90, 94.48, 95.14, 95.56, 95.95, 
    96.26, 96.47, 96.79, 96.93, 97.11, 97.21, 97.41, 97.49, 97.71, 97.73, 
    97.91, 98.05, 98.04, 98.19, 98.22, 98.26, 98.42, 98.46, 98.53, 98.53, 
    98.61, 98.64, 98.73, 98.69, 98.81, 98.83, 98.77, 98.88, 98.94, 98.87, 
    99.01, 99.04, 99.00, 99.07, 99.08, 99.08, 99.11, 99.12, 99.14, 99.20, 
    99.19, 99.20, 99.17, 99.19, 99.22, 99.28, 99.25, 99.27, 99.33, 99.28, 
    99.28, 99.37, 99.32, 99.40, 99.38, 99.36, 99.40, 99.35, 99.38, 99.39, 
    99.40, 99.41, 99.39, 99.48, 99.45, 99.42, 99.42, 99.44, 99.48, 99.44, 
    99.46, 99.46, 99.45, 99.46, 99.50, 99.54, 99.54, 99.50, 99.52, 99.50, 
    99.52, 99.50, 99.54, 99.51, 99.54, 99.58, 99.54, 99.54, 99.56, 99.54, 
    99.57, 99.55, 99.51, 99.58, 99.57, 99.57, 99.56, 99.60, 99.55, 99.62, 
    99.60, 99.62, 99.59, 99.60, 99.59, 99.64, 99.64, 99.63, 99.65, 99.66, 
    99.57, 99.64, 99.60, 99.65, 99.61, 99.62, 99.66, 99.63
]

# Create the Loss plot
p_loss = figure(title="Loss Over Epochs (Interactive)", x_axis_label="Epochs", y_axis_label="Loss", width=700, height=400)
p_loss.add_tools(HoverTool(tooltips=[("Epoch", "$x"), ("Loss", "$y")]))
p_loss.line(epochs_5_no_cuda, loss_5_no_cuda, legend_label="No CUDA (5 Iterations)", line_width=2, color="blue")
p_loss.circle(epochs_5_no_cuda, loss_5_no_cuda, color="blue", size=6)
p_loss.line(epochs_5_cuda_v1, loss_5_cuda_v1, legend_label="CUDA (5 Iterations), v1", line_width=2, color="green")
p_loss.circle(epochs_5_cuda_v1, loss_5_cuda_v1, color="green", size=6)
p_loss.line(epochs_10_cuda_v1, loss_10_cuda_v1, legend_label="CUDA (10 Iterations, v1)", line_width=2, color="red")
p_loss.circle(epochs_10_cuda_v1, loss_10_cuda_v1, color="red", size=6)
p_loss.line(epochs_10_cuda_v2, loss_10_cuda_v2, legend_label="CUDA (10 Iterations, v2)", line_width=2, color="purple")
p_loss.circle(epochs_10_cuda_v2, loss_10_cuda_v2, color="purple", size=6)
p_loss.line(epochs_128_cuda, loss_128_cuda, legend_label="CUDA (128 Iterations)", line_width=2, color="orange")
p_loss.circle(epochs_128_cuda, loss_128_cuda, color="orange", size=6)

# Configure the legend for the loss plot
p_loss.legend.title = "Configurations"
p_loss.legend.location = "center_right"
p_loss.legend.click_policy = "hide"  # Allows toggling visibility

# Create the Accuracy plot
p_accuracy = figure(title="Accuracy Over Epochs (Interactive)", x_axis_label="Epochs", y_axis_label="Accuracy (%)", width=700, height=400)
p_accuracy.add_tools(HoverTool(tooltips=[("Epoch", "$x"), ("Accuracy", "$y")]))
p_accuracy.line(epochs_5_no_cuda, accuracy_5_no_cuda, legend_label="No CUDA (5 Iterations)", line_width=2, color="blue")
p_accuracy.circle(epochs_5_no_cuda, accuracy_5_no_cuda, color="blue", size=6)
p_accuracy.line(epochs_5_cuda_v1, accuracy_5_cuda_v1, legend_label="CUDA (5 Iterations), v1", line_width=2, color="green")
p_accuracy.circle(epochs_5_cuda_v1, accuracy_5_cuda_v1, color="green", size=6)
p_accuracy.line(epochs_10_cuda_v1, accuracy_10_cuda_v1, legend_label="CUDA (10 Iterations, v1)", line_width=2, color="red")
p_accuracy.circle(epochs_10_cuda_v1, accuracy_10_cuda_v1, color="red", size=6)
p_accuracy.line(epochs_10_cuda_v2, accuracy_10_cuda_v2, legend_label="CUDA (10 Iterations, v2)", line_width=2, color="purple")
p_accuracy.circle(epochs_10_cuda_v2, accuracy_10_cuda_v2, color="purple", size=6)
p_accuracy.line(epochs_128_cuda, accuracy_128_cuda, legend_label="CUDA (128 Iterations)", line_width=2, color="orange")
p_accuracy.circle(epochs_128_cuda, accuracy_128_cuda, color="orange", size=6)

# Configure the legend for the accuracy plot
p_accuracy.legend.title = "Configurations"
p_accuracy.legend.location = "center_right"
p_accuracy.legend.click_policy = "hide"  # Allows toggling visibility

# Display both plots in a column layout
show(column(p_loss, p_accuracy))