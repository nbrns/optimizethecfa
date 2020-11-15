"""Module with extensions to export setups, plots, etc."""
import matplotlib
import matplotlib.pyplot as plt
import torch


def export_setup(model, loss_function, label_mapping, logger):
    logger.info(f'Model:\n{model}')
    logger.info(f'Loss function:\n{loss_function}')
    logger.info(f'Label mapping:\n{label_mapping}')


def export_plots(path, loss, prediction):
    # export loss
    loss_values = [l[1] for l in loss]
    plt.plot(loss_values)
    plt.savefig(f'{path}/loss.png')
    plt.clf()


    # export result
    # set up colors
    colors = ['#000000', '#ff0000', '#00ff00',
              '#0000ff', '#ff00ff', '#ffff00',
              '#00ffff', '#880000', '#008800',
              '#000088', '#888800', '#008888',
              '#880088', '#ffffff']

    # plot result
    result = torch.argmax(prediction, dim=1).cpu().detach().numpy()
    plt.figure(figsize=(15, 10))
    plt.imshow(result.squeeze(), cmap=matplotlib.colors.ListedColormap(colors))
    plt.colorbar()
    plt.savefig(f'{path}/prediction.png')


def export_prediction(path, prediction):
    plt.ioff()
    result = torch.argmax(prediction, dim=1).cpu().detach().numpy()
    colors = ['#000000', '#ff0000', '#00ff00',
              '#0000ff', '#ff00ff', '#ffff00',
              '#00ffff', '#880000', '#008800',
              '#000088', '#888800', '#008888',
              '#880088', '#ffffff']
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(result.squeeze(), cmap=matplotlib.colors.ListedColormap(colors))
    plt.colorbar()
    plt.savefig(f'{path}')
    plt.close(fig)
