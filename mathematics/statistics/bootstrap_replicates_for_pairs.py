# Ideas from : https://campus.datacamp.com/courses/statistical-thinking-in-python-part-2/bootstrap-confidence-intervals?ex=12

import numpy as np
import matplotlib.pyplot as plt


def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps


def show_confidence_intervals(data, data_name):
    # Predict 95% Confidence interval (CI) for data
    ci_data = np.percentile(data, [2.5, 97.5])

    print(f"Confidence Interval for {data_name} {ci_data[0]}"
          f" to {ci_data[1]}")

    # Plot the histogram
    hist_data = plt.hist(data, bins=50, density=True)
    print(hist_data)
    _ = plt.xlabel(data_name)
    _ = plt.ylabel('PDF')
    plt.title(f"Bootstrap Replicates of {data_name}")
    # Note we are using axvline instead of vlines
    # See https://stackohttps://stackoverflow.com/questions/24988448/how-to-draw-vertical-lines-on-a-given-plotverflow.com/questions/24988448/how-to-draw-vertical-lines-on-a-given-plot
    plt.axvline(ci_data[0], color="r")
    plt.axvline(ci_data[1], color="r")
    plt.show()


# Pretend that first_fake_data is number of votes per county
# and second_fake_data is percentage votes for Obama
# That is, pretend our data comes in "pairs", and that
# we want to resample them as pairs.
first_fake_data = np.arange(0, 101, 1)
second_fake_data = (2 + np.random.random(size=len(first_fake_data))) * \
    first_fake_data + 4 + \
    21 * np.random.random(size=len(first_fake_data))


plt.plot(first_fake_data, second_fake_data)
plt.title("Original Data")
plt.show()

bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(
    first_fake_data, second_fake_data, size=1000)

show_confidence_intervals(bs_slope_reps, "Slope")
show_confidence_intervals(bs_intercept_reps, "Intercept")


# Generate array of x-values for bootstrap lines: x
x = np.array([0, 100])

# Plot the bootstrap lines
for i in range(1000):
    _ = plt.plot(x,
                 bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(first_fake_data, second_fake_data, marker=".", linestyle="none")

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()
