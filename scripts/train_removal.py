import sys, torch, pandas as pd
from vpm.export import load_generator
from vpm.removal import train_removal
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
m = pd.read_csv("data/manifests/all.csv"); g = load_generator(sys.argv[1])
net, hist = train_removal(g, m[m.split == "train"].clean_path.tolist())
torch.save(net.state_dict(), sys.argv[2]); pd.DataFrame(dict(loss=hist)).to_csv(sys.argv[2] + ".log.csv", index=False)
