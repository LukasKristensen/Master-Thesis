import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


vec_quant_sweep = pd.read_csv("C:\\Users\\lukad\\Desktop\\CE10\\wandb exports\\param_sweep_vectorquant_2024-05-06T13_11_14.085+02_00.csv")
print(vec_quant_sweep.columns)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
vec_quant_sweep = vec_quant_sweep.groupby(['batch_size', 'learning_rate', 'commitment_cost', 'num_latents', 'epochs']).mean().reset_index()

surf = ax.plot_trisurf(vec_quant_sweep['commitment_cost'], vec_quant_sweep['num_latents'], vec_quant_sweep['val_loss'], cmap='viridis', edgecolor='none')
ax.set_xlabel('commitment_cost')
ax.set_ylabel('num_latents')
ax.set_zlabel('val_loss')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

for index in vec_quant_sweep.columns:
    if index == 'val_loss':
        continue
    index_data = vec_quant_sweep.groupby(index).mean().reset_index()
    plt.plot(index_data[index], index_data['val_loss'])
    plt.xlabel(index)
    plt.ylabel('val_loss')
    plt.show()
    print("Index: ", index)
    print(index_data)

vec_quant_sweep = vec_quant_sweep.sort_values(by='val_loss')
print(vec_quant_sweep)

batch_learning_sweep = pd.read_csv("C:\\Users\\lukad\\Desktop\\CE10\\wandb exports\\param_sweep_batch_learning_2024-05-06T13_11_14.085+02_00.csv")
print(batch_learning_sweep.columns)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

batch_learning_sweep = batch_learning_sweep.groupby(['batch_size', 'learning_rate', 'commitment_cost', 'num_latents', 'epochs']).mean().reset_index()
log_learning_rate = np.log10(batch_learning_sweep['learning_rate'])

surf = ax.plot_trisurf(log_learning_rate, batch_learning_sweep['batch_size'], batch_learning_sweep['val_loss'], cmap='viridis', edgecolor='none')
ax.set_xlabel('log(learning_rate)')
ax.set_ylabel('batch_size')
ax.set_zlabel('val_loss')

# Change the x-ticks labels to represent the original learning_rate values
ax.set_xticks(np.log10([0.00001, 0.0001, 0.001, 0.01]))
ax.set_xticklabels([0.00001, 0.0001, 0.001, 0.01])

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


packet_loss_sweep = pd.read_csv("C:\\Users\\lukad\\Desktop\\CE10\\wandb exports\\PACKET_LOSS_2024-05-07T04_42_28.072+02_00.csv")
packet_loss = packet_loss_sweep['packet_loss_percentage']
val_loss = packet_loss_sweep['val_loss']

plt.plot(packet_loss, val_loss, marker='o')
plt.xlabel('packet_loss')
plt.ylabel('val_loss')
plt.grid(True)
plt.show()