import matplotlib as mpl
import matplotlib.cm as cm

norm = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = cm.hot
x = 5

m = cm.ScalarMappable( cmap=cmap)
print(m.to_rgba(x))