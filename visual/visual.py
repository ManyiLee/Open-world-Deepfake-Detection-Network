import numpy as np  
import pickle  
import matplotlib.pyplot as plt  
from sklearn.manifold import TSNE

def V3D_feature(feat_dict, dataset, data, ev, az ,name):

    feat = feat_dict[dataset][data]
    t_feat, t_cmp_feat, f_feat, f_cmp_feat = feat["t_feat"], feat["t_cmp_feat"], feat["f_feat"], feat["f_cmp_feat"]

    proj = np.concatenate((t_feat, t_cmp_feat, f_feat, f_cmp_feat), axis=0)
    tsne = TSNE(n_components=3, learning_rate=200, n_iter=2000, random_state=200371)
    X_tsne = tsne.fit_transform(proj)

    colors = ['#3BCFF0','#007FDB','#E664D7','#FFFF00']   
    markers = ['o','*','o','*']  
    sizes = [15, 15, 15, 15]
            

    plt.figure(figsize=(24, 18))
    ax = plt.axes(projection="3d")
    ax.view_init(elev=ev, azim=az)
    ax.grid(False)
    ax.axis('off') 

    pos = [X_tsne[:len(t_feat)], X_tsne[len(t_feat):len(t_feat) + len(t_cmp_feat)], \
            X_tsne[len(t_feat) + len(t_cmp_feat): len(t_feat) + len(t_cmp_feat)+ len(f_feat)], \
            X_tsne[len(t_feat) + len(t_cmp_feat)+ len(f_feat):]]
    for p, color, mark, size in zip(pos, colors, markers, sizes):  
        ax.scatter3D(p[:, 0], p[:, 1], p[:, 2],c=color, marker=mark, s=50)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])
    plt.savefig("./visual/{}-{}-3D-{}-{}-{}.png".format(dataset,data, ev, az,name), dpi=300)

def V2D_feature(feat_dict, dataset, data, name):

    feat = feat_dict[dataset][data]
    t_feat, t_cmp_feat, f_feat, f_cmp_feat = feat["t_feat"], feat["t_cmp_feat"], feat["f_feat"], feat["f_cmp_feat"]
    """
    proj = np.concatenate((t_feat, t_cmp_feat, f_feat, f_cmp_feat), axis=0)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=2000, random_state=200371)
    X_tsne = tsne.fit_transform(proj)
    """
    colors = ['#3BCFF0','#007FDB','#FF1F00','#FF7246',]   
    markers = ['o','x','o','x']
    sizes = [15,15,15,15]
            
    plt.figure(figsize=(8, 6))
    """
    pos = [X_tsne[:len(t_feat)], X_tsne[len(t_feat):len(t_feat) + len(t_cmp_feat)], \
            X_tsne[len(t_feat) + len(t_cmp_feat): len(t_feat) + len(t_cmp_feat)+ len(f_feat)], \
            X_tsne[len(t_feat) + len(t_cmp_feat)+ len(f_feat):]]
    """
    for proj, color, mark, size in zip([t_feat, t_cmp_feat, f_feat, f_cmp_feat], colors, markers, sizes):
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=2000, random_state=200371)
        X_tsne = tsne.fit_transform(proj)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=color, marker=mark, s=size) 

    plt.savefig("./visual/{}-{}-2D-{}.png".format(dataset,data, name), dpi=300)

if __name__ == "__main__":
    #test_mygen9GANs ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
    #test_8GANs ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
    # 加载数组  
    with open('./image/0-feat.pkl', 'rb') as file:  
        feat_dict = pickle.load(file)  

    #V2D_feature(feat_dict,"test_8GANs", "cyclegan", "baseline")
    V3D_feature(feat_dict, "test_8GANs", "cyclegan", -90, 0, "baseline")
    

    # 加载数组  
    with open('./image/27-feat.pkl', 'rb') as file:  
        feat_dict = pickle.load(file)  

    #V2D_feature(feat_dict,"test_8GANs", "cyclegan", "ours")
    V3D_feature(feat_dict, "test_8GANs", "cyclegan", -90, 0, "ours")
