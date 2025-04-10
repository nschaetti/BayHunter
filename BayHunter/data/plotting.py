#
# MIGRATE
#

# Imports
import matplotlib.pyplot as plt


def plot_models(
    models,
    labels=None,
    colors=None,
    invert_axes=False,
    title="Seismic Models",
    show=True
):
    """
    Plot multiple SeismicModel instances on the same axes.

    :param models: List of SeismicModel objects
    :param labels: Optional list of labels
    :param colors: Optional list of colors
    :param invert_axes: Invert axes (default: False)
    :param title: Plot title
    :param show: Whether to call plt.show()
    :return: Matplotlib Axes
    """
    fig, ax = plt.subplots()
    for i, model in enumerate(models):
        label = labels[i] if labels else None
        color = colors[i] if colors else None
        model.plot(ax=ax, label=label, color=color, invert_axes=invert_axes)
    # end for

    ax.set_title(title)
    if labels:
        ax.legend()
    if show:
        plt.show()
    # end if
    return ax
# end plot_models