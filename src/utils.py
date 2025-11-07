import numpy as np
import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_reliability_diagram(samples, true_labels, M=10, framework='numpy'):
    """
    Generates a reliability diagram.

    Args:
        samples: Predicted probabilities for each class.
        true_labels: True labels for each sample.
        M: Number of bins to use for calibration.
        framework: The framework used for the samples ('numpy', 'torch', or 'jax').
    """
    if framework == 'numpy':
        confidences = np.max(samples, axis=1)
        predicted_labels = np.argmax(samples, axis=1)
        accuracies = predicted_labels == true_labels
        bin_boundaries = np.linspace(0, 1, M + 1)
    elif framework == 'jax':
        confidences = jnp.max(samples, axis=1)
        predicted_labels = jnp.argmax(samples, axis=1)
        accuracies = predicted_labels == true_labels
        bin_boundaries = jnp.linspace(0, 1, M + 1)
    else:
        raise ValueError("Framework must be 'numpy', 'torch', or 'jax'")


    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    avg_confidences = []
    bin_accuracies = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if framework == 'numpy':
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        elif framework == 'jax':
            in_bin = jnp.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())

        if in_bin.sum() > 0:
            if framework == 'numpy':
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                accuracy_in_bin = np.mean(accuracies[in_bin])
            elif framework == 'jax':
                avg_confidence_in_bin = jnp.mean(confidences[in_bin]).item()
                accuracy_in_bin = jnp.mean(accuracies[in_bin].astype(jnp.float32)).item()

            avg_confidences.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
        else:
            # Append 0 if no samples in bin to maintain plot structure
            avg_confidences.append(0)
            bin_accuracies.append(0)


    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    # Filter out zero values for plotting if necessary, or handle in plotting
    plt.plot(avg_confidences, bin_accuracies, marker='o', linestyle='-', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.show()

def expected_calibration_error_jax(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = jnp.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = jnp.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = jnp.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label == true_labels

    ece = jnp.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = jnp.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = jnp.mean(in_bin.astype(jnp.float32))

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = jnp.mean(accuracies[in_bin].astype(jnp.float32))
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = jnp.mean(confidences[in_bin])
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece

def quantization(x, s, z, alpha_q, beta_q):
    x_q = jnp.round(1 / s * x + z, decimals=0)
    x_q = jnp.clip(x_q, a_min=alpha_q, a_max=beta_q)
    return x_q.astype(jnp.uint8)


def quantization_int8(x, s, z):
    x_q = quantization(x, s, z, alpha_q=-128, beta_q=127)
    x_q = x_q.astype(jnp.int8)
    return x_q

def dequantization(x_q, s, z):
    # x_q - z might go outside the quantization range.
    x_q = x_q.astype(jnp.int32)
    x = s * (x_q - z)
    x = x.astype(jnp.float32)
    return x


def generate_quantization_constants_scale(alpha, beta, alpha_q, beta_q):
    # Affine quantization mapping
    s = (beta - alpha) / (beta_q - alpha_q)
    return s

def generate_quantization_constants_bias(alpha, beta, alpha_q, beta_q):
    # Affine quantization mapping
    z = jnp.int8((beta * alpha_q - alpha * beta_q) / (beta - alpha))
    return z

def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]