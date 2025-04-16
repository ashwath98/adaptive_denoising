import torch
import wandb
from src.sde.implementations import VPSDE
from src.models.score_model import FullConnectedScoreModel
from src.training.trainer import DiffusionTrainer
#from src.utils.visualization import plot_score_vector_field
from src.data.manifold_dataset import ManifoldDataset
from src.data.manifold_function import return_data, plot_data, manifold_func
from src.utils.visualization import DiffusionVisualizer
import matplotlib.pyplot as plt
from backward import run_backwards

def generate_backward_visualization(output):
    
    from celluloid import Camera # getting the camera
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import HTML # to show the animation in Jupyter
# the camera gets the fig we'll plot

    fig, axs = plt.subplots(figsize=(7,7))
    camera = Camera(fig)
    x_ref = np.linspace(0,1,100)
    for idx in range(output.shape[0]):
        axs.plot(x_ref, manifold_func(x_ref),color='tab:red')
        axs.scatter(output[idx,:,0],output[idx,:,1],color='tab:blue')
        axs.set_xlim(-2.0,2.0)
        axs.set_ylim(-2.0,2.0)
        camera.snap()

    animation = camera.animate() # animation ready
    animation.save('diffusion_model_sampling.gif')
def visualize_forward_process(data, sde):
    n_grid_points = 8
    time_vec = torch.linspace(0,1,n_grid_points)**2
    X_0 = torch.stack([torch.stack([data[idx]]*n_grid_points) for idx in range(1000)]).transpose(1,0)
    X_t, noise, score = sde.run_forward(X_0,time_vec)
    fig, axs = plt.subplots(1,n_grid_points, figsize=(6*n_grid_points,6))
    for idx in range(n_grid_points):
        axs[idx].scatter(X_t[idx,:,0],X_t[idx,:,1])
        axs[idx].set_xlim(-1.5,1.5)
        axs[idx].set_ylim(-1.5,1.5)
        axs[idx].set_title(f"time step = {time_vec[idx]:.2f}")
    plt.show()
def main():
    # Configuration
    config = {
        'learning_rate': 2e-5,
        'weight_decay': 1e-5,
        'n_epochs': 10,
        'batch_size': 128,
        'train_score': False,
        'log_every': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sde_params': {
            'T_max': 1.0,
            'beta_min': 0.01,
            'beta_max': 2.0
        }
    }

    # Initialize wandb
    wandb.init(project="diffusion-experiments", config=config)

    # Initialize components
    sde = VPSDE(**config['sde_params'])
    model = FullConnectedScoreModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Create your dataset and dataloader here
    # dataset = YourDataset()
    # dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    N_SAMPLES = 50000
    data = return_data()
    print(data.shape)
    dataset = ManifoldDataset(data)
    batch_size = 128
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize trainer
    visualizer = DiffusionVisualizer(sde, model, device=config['device'])

    # Visualize original data
    # visualizer.plot_data_manifold(
    #     data=dataset.data,
    #     manifold_func=None,
    #     save_path="plots/original_data.png"
    # )
    plot_data(data)
    visualize_forward_process(data, sde)

    
    # Visualize forward process
    # visualizer.visualize_forward_process(
    #     initial_samples=dataset.data[:1000],
    #     save_path="plots/forward_process.png"
    # )
    #import pdb; pdb.set_trace()
    trainer = DiffusionTrainer(
        model=model,
        sde=sde,
        dataloader=dataloader,
        optimizer=optimizer,
        device=config['device'],
        config=config
    )

    # Training loop
    for epoch in range(config['n_epochs']):
        loss = trainer.train_epoch(epoch)
        #wandb.log({"loss": loss})
        # Visualization
        # if (epoch + 1) % 5 == 0:
        #     for t in [0.0, 0.25, 0.5, 1.0]:
        #         fig = plot_score_vector_field(
        #             model=model,
        #             t=t,
        #             sde=sde,
        #             train_score=config['train_score'],
        #             save_path=f"plots/vector_field_epoch_{epoch}_t_{t}.png"
        #         )
        #         wandb.log({f"vector_field_t_{t}": wandb.Image(fig)})

    x_start = torch.randn(size=next(enumerate(dataloader))[1].shape)
    output,time_grid = run_backwards(model,sde,x_start=x_start,n_steps=10,device=config['device'], train_score=config['train_score'])
    generate_backward_visualization(output)

if __name__ == "__main__":
    main() 