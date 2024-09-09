from pathlib import Path

import epydeck
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import Markdown, display
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)  # noqa: NPY002


def print_diff(training_list, testing_list, returned_list):
    diff_output = []

    for item in returned_list:
        if item in training_list:
            diff_output.append(f"- {item} (from training list)")
        elif item in testing_list:
            diff_output.append(f"+ {item} (from testing list)")
        else:
            diff_output.append(f"  {item} (not in training or testing list)")

    # Join the diff output and display it as markdown for better formatting
    display(Markdown(f"```\n{'\n'.join(diff_output)}\n```"))


def calculate_eed(ds: xr.Dataset) -> None:
    ds["Electron_Energy_Distribution"] = (
        ds["Derived/Average_Particle_Energy/Electron"]
        * ds["Derived/Number_Density/Electron"]
    )

    # metadata
    ds["Electron_Energy_Distribution"].attrs[
        "long_name"
    ] = "Electron Energy Distribution"
    ds["Electron_Energy_Distribution"].attrs["units"] = "J/cc"


def display_plots(gpr, X_scaled, scaler, X_train, y_train):
    # Generate a grid of values over the input space
    x1 = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
    x2 = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.array([X1.ravel(), X2.ravel()]).T

    # Scale the grid points using the fitted scaler
    X_grid_scaled = scaler.transform(X_grid)

    # Predict the mean and standard deviation over the grid
    mean, std = gpr.predict(X_grid_scaled, return_std=True)

    # Reshape the results back to the grid shape
    mean = mean.reshape(X1.shape)
    std = std.reshape(X1.shape)

    # Plotting the mean prediction
    # This plot shows the mean prediction of the GPR model across the input space. The surface represents the predicted electron energy distribution based on the input features intensity and density. Red dots represent your training data points.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, mean, cmap="viridis", alpha=0.7)
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color="r", label="Training data")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_zlabel("Predicted Electron Energy Distribution")
    ax.set_title("Gaussian Process Regression Mean Prediction")
    plt.legend()
    plt.show()

    # Plotting the standard deviation (uncertainty)
    # This plot visualizes the uncertainty (standard deviation) of the predictions. Areas with higher uncertainty indicate that the model is less confident in its predictions, possibly due to fewer nearby training points.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, std, cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_zlabel("Uncertainty (Std Dev)")
    ax.set_title("Gaussian Process Regression Uncertainty")
    plt.show()

    # Calculate confidence intervals
    upper_confidence = mean + 1.96 * std
    lower_confidence = mean - 1.96 * std

    # Plotting the confidence intervals
    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, mean, levels=100, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Mean prediction")
    plt.contour(
        X1, X2, upper_confidence, levels=10, colors="orange", alpha=0.5, linestyles="--"
    )
    plt.contour(
        X1, X2, lower_confidence, levels=10, colors="orange", alpha=0.5, linestyles="--"
    )
    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=y.to_numpy().ravel(),
        cmap="viridis",
        edgecolors="k",
        marker="o",
        label="Data points",
    )
    plt.title("GPR Mean Prediction with Confidence Intervals")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

    # Draw samples from the GP posterior
    samples = gpr.sample_y(X_grid_scaled, n_samples=5)

    # Plotting the samples
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, sample in enumerate(samples.T):
        ax.plot_surface(
            X1, X2, sample.reshape(X1.shape), alpha=0.3, label=f"Sample {i+1}"
        )
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color="r", label="Training data")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_zlabel("Sampled Functions")
    ax.set_title("Samples from the GP Posterior")
    plt.legend()
    plt.show()


def plot_actual_v_predicted(  # noqa: PLR0913
    gpr: GaussianProcessRegressor,
    X: np.ndarray,
    y: np.ndarray,
    attribute: str,
    units: str,
    highest_y_uncertainty_indices: np.ndarray = None,
):
    y_predicted, std = gpr.predict(X, return_std=True)
    plt.figure()
    plt.plot(y, linestyle="-", label="Actual")
    plt.plot(y_predicted, linestyle="--", label="Predicted")

    # Plot the highest uncertainty points
    plt.fill_between(
        np.arange(len(y)),
        y_predicted - std,
        y_predicted + std,
        alpha=0.3,
        color="gray",
        label="Uncertainty",
    )
    # Highlight the points with the highest uncertainty
    if highest_y_uncertainty_indices is not None:
        plt.scatter(
            highest_y_uncertainty_indices,
            y[highest_y_uncertainty_indices],
            color="red",
            label="High Uncertainty",
            marker="o",
            s=100,  # size of the markers
        )

    plt.title(f"Gaussian process regression, R2={gpr.score(X, y):.2f}")
    plt.xlabel("Index")
    plt.ylabel(f"{attribute} [${units}$]")
    plt.legend()
    plt.show()


def highest_uncertainty_indices(
    gpr: GaussianProcessRegressor, X: np.ndarray, n_points: int
) -> np.ndarray:
    _, std = gpr.predict(X, return_std=True)

    # Find the indices of the points with the highest uncertainty
    return np.argsort(std)[-n_points:][::-1]


# TODO: Fix this hack
script_path = Path("/Users/joel/Source/epoch_runner")
simulation_path_filename = "paths.txt"

# Load in the latest simulation runs
with Path.open(script_path / simulation_path_filename) as f:
    paths = [Path(path) for path in f.read().strip().split("\n")]


# Load in the feature matrix X from decks
# ---------------------------------------
X = {"intensity": [], "density": []}

for path in paths:
    with Path.open(path / "input.deck") as f:
        deck = epydeck.load(f)
        X["intensity"].append(deck["constant"]["intens"])
        X["density"].append(deck["constant"]["nel"])

X = pd.DataFrame(X)

# Load in the target values y from sdf
# ------------------------------------
y = {"Electron_Energy_Distribution_max_x": []}

for path in paths:
    df_80 = xr.open_dataset(path / "0080.sdf")
    calculate_eed(df_80)
    electron_energy_distribution = df_80["Electron_Energy_Distribution"]
    x_coord_name = "X_Grid_mid"
    y_coord_name = "Y_Grid_mid"
    flat_index = np.argmax(
        electron_energy_distribution.values
    )  # Get the flat index of the maximum value

    # Convert the flat index to multi-dimensional indices
    max_indices = np.unravel_index(flat_index, electron_energy_distribution.shape)

    # Retrieve the coordinates corresponding to the indices
    max_x_coord = electron_energy_distribution.coords[x_coord_name].to_numpy()[
        max_indices[0]
    ]
    # max_y_coord = electron_energy_distribution.coords[y_coord_name].to_numpy()[max_indices[1]]

    y["Electron_Energy_Distribution_max_x"].append(max_x_coord)

y = pd.DataFrame(y)

# display(pd.concat([X, y], keys=["X features", "y target values"], axis=1))


# Training model
# --------------
scaler = MinMaxScaler(feature_range=(-5, 5)).fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, np.ravel(y))

# lentgh_scale determines the reach of influence on neighboring points
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + ConstantKernel(10)
# kernel = 1 * Matern(length_scale=1.0, nu=2.5) + ConstantKernel(constant_value=10)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, random_state=0)
gpr.fit(X_train, y_train)

# print(gpr.score(X_test, y_test))

# Highest uncertainty on all data
plot_actual_v_predicted(
    gpr,
    X_scaled,
    y["Electron_Energy_Distribution_max_x"],
    "Electron Energy Distribution max x",
    "J/cc",
)

# Highest uncertainty on test data
plot_actual_v_predicted(
    gpr, X_test, y_test, "Electron Energy Distribution max x", "J/cc"
)

# Highest uncertainty on training data
highest_y_uncertainty_indices = highest_uncertainty_indices(gpr, X_scaled, 5)
highest_X_uncertainty = scaler.inverse_transform(
    X_scaled[highest_y_uncertainty_indices]
)
# This is proof that the model is fitting the training data well
# as all the most uncertain points aren't solely in the test data
print_diff(
    scaler.inverse_transform(X_train),
    scaler.inverse_transform(X_test),
    highest_X_uncertainty,
)

plot_actual_v_predicted(
    gpr,
    X_scaled,
    y["Electron_Energy_Distribution_max_x"],
    "Electron Energy Distribution max x",
    "J/cc",
    highest_y_uncertainty_indices,
)
