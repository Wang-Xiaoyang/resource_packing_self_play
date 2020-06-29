import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colorbar as cbar

numAxises = 2 # 2D
numItems = 2
binWidth, binHeight = 10, 10
np.random.seed(100)

item_list = [[binWidth, binHeight, 0, 0]] # initial item equals to the bin

while len(item_list) < numItems:
    axis = np.random.randint(2) # 0 for x , 1 for y axis
    idx_item = np.random.randint(len(item_list)) # choose an item to split
    [w, h, a, b] = item_list[idx_item]
    if axis == 0:
        if w == 1:
            continue
        x_split = np.random.randint(a+1, a+w)
        new_w = x_split - a
        item_s1 = [new_w, h, a, b]
        item_list.append(item_s1)
        item_s2 = [w-new_w, h, x_split, b]
        item_list.append(item_s2)
        item_list.pop(idx_item)
    elif axis == 1:
        if h == 1:
            continue
        y_split = np.random.randint(b+1, b+h)
        new_h = y_split - b
        item_s1 = [w, new_h, a, b]
        item_list.append(item_s1)
        item_s2 = [w, h-new_h, a, y_split]
        item_list.append(item_s2)
        item_list.pop(idx_item)

# display
fig,ax=plt.subplots(1)
plt.ylim(0,binHeight)
plt.xlim(0,binWidth)
colors=np.random.rand(numItems)
cmap=plt.cm.RdYlBu_r
c=cmap(colors)

for i in range(numItems):
    [w, h, a, b] = item_list[i]
    rect=patches.Rectangle((a, b), w, h,
                            edgecolor='black',
                            linewidth = 0.5,
                            facecolor = c[i],
                            )
    ax.add_patch(rect)

plt.show()