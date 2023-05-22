from re import X
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import matplotlib.cm as cm

def save_causal_graph(save_path, causal_matrix: np.ndarray, thres_percentile=100, colormap="gnuplot2"):
    causal_matrix = np.max(causal_matrix)
    print(causal_matrix.shape)
    
    n_node = causal_matrix.shape[0]
    image_size = [100, 100]
    n_node_dim = n_node**0.5
    causal_thres = np.percentile(causal_matrix, 100-thres_percentile)
    
    colormap = cm.get_cmap(colormap)

    plt.figure(figsize=[10,10], facecolor='black', edgecolor='black')

        
    for node_i_from in tqdm.tqdm(range(n_node)):
        x_from = image_size[0] // n_node_dim * (node_i_from // n_node_dim + 0.5)
        y_from = image_size[0] // n_node_dim * (node_i_from % n_node_dim + 0.5)
        plt.text(y=x_from, x=y_from, s=f"{node_i_from:d}", size=20, color="#ffffff")
        for node_i_to in range(n_node):
            if not node_i_from == node_i_to:
                x_to = image_size[0] // n_node_dim * (node_i_to // n_node_dim + 0.5)
                y_to = image_size[0] // n_node_dim * (node_i_to % n_node_dim + 0.5)
                causal_effect = causal_matrix[node_i_from, node_i_to]
                if causal_effect > causal_thres:
                    width = max(0.01, 1*causal_effect)
                    arrow_length = ((x_to-x_from)**2 + (y_to-y_from)**2)**0.5
                    plt.arrow(
                        y=x_from+(x_to-x_from)*width/arrow_length,
                        x=y_from+(y_to-y_from)*width/arrow_length,
                        dy=(x_to-x_from)*(arrow_length-5*width)/arrow_length,
                        dx=(y_to-y_from)*(arrow_length-5*width)/arrow_length,
                        width=width,
                        head_length=4*width,
                        facecolor=colormap(causal_effect)[:3]+(causal_effect,),
                        edgecolor="#00000000"
                    )
        
        # fig.add_annotation(text=f"{node_i_from:d}", x=x_from, y=y_from, showarrow=False
    
    ax=plt.gca()
    ax.patch.set_facecolor("black")
    ax.xaxis.set_ticks_position('top') 
    ax.invert_yaxis()
    plt.savefig(save_path)
    
    
if __name__=="__main__":
    causal_matrix = np.load("outputs/tsgae_2022_0716_191203_262072/w.npy")
    save_causal_graph("outputs/pic/causal.png", causal_matrix, thres_percentile=100)