#!/usr/bin/env python
# coding: utf-8

# In[2]:


import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt

# Download the OSM road network data
G = ox.graph_from_place("Bangaluru")

# Convert the road network to a binary image
fig, ax = ox.plot_graph(G, show=False, close=True, node_size=0)
img = np.array(fig.canvas.renderer._renderer)
img_gray = img.mean(axis=2)
img_binary = img_gray > 100

# Apply the box counting method
box_sizes = range(1, 200)
box_counts = []
road_counts = []
boxes_cover = []

for box_size in box_sizes:
    count_boxes = 0
    count_roads = 0
    count_cover = 0
    for i in range(0, img_binary.shape[0], box_size):
        for j in range(0, img_binary.shape[1], box_size):
            count_boxes += 1
            if img_binary[i:i+box_size, j:j+box_size].any():
                count_roads += 1
                count_cover += 1
            else:
                count_cover += 0
    box_counts.append(count_boxes)
    road_counts.append(count_roads)
    boxes_cover.append(count_cover)

# Calculate the fractal dimension
slope, intercept = np.polyfit(np.log(box_counts), np.log(road_counts), 1)
fractal_dimension = -slope

# Plot the results
fig, ax = plt.subplots()
ax.scatter(box_counts, road_counts)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Box size')
ax.set_ylabel('Number of boxes with road pixels')
plt.show()

print(f"The fractal dimension of the road network is: {fractal_dimension}")
print(f"The list of the number of boxes that cover the road network is: {boxes_cover}")
print(f"The total number of boxes is: {count_boxes}")
for i in range(len(box_sizes)):
    print(f"Box size: {box_sizes[i]}, number of boxes: {box_counts[i]},number of boxes covered:{boxes_cover[i]}")


# In[4]:


import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt

# Download the OSM road network data
G = ox.graph_from_place("Shopian")

# Convert the road network to a binary image
fig, ax = ox.plot_graph(G, show=False, close=True, node_size=0)
img = np.array(fig.canvas.renderer._renderer)
img_gray = img.mean(axis=2)
img_binary = img_gray > 100

# Apply the box counting method
box_sizes = range(1, 200)
box_counts = []
road_counts = []
boxes_cover = []

for box_size in box_sizes:
    count_boxes = 0
    count_roads = 0
    count_cover = 0
    for i in range(0, img_binary.shape[0], box_size):
        for j in range(0, img_binary.shape[1], box_size):
            count_boxes += 1
            if img_binary[i:i+box_size, j:j+box_size].any():
                count_roads += 1
                count_cover += 1
            else:
                count_cover += 0
    box_counts.append(count_boxes)
    road_counts.append(count_roads)
    boxes_cover.append(count_cover)



# In[ ]:





# In[8]:


# Calculate the fractal dimension
slope, intercept = np.polyfit(np.log(box_counts), np.log(road_counts), 1)
fractal_dimension = slope

# Plot the results
fig, ax = plt.subplots()
ax.scatter(box_sizes,boxes_cover)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Box size')
ax.set_ylabel('Number of boxes with road pixels')
plt.show()

print(f"The fractal dimension of the road network is: {fractal_dimension}")
print(f"The total number of boxes is: {count_boxes}")
for i in range(len(box_sizes)):
    print(f"Box size: {box_sizes[i]}, number of boxes: {box_counts[i]},number of boxes covered:{boxes_cover[i]}")


# In[ ]:





# In[ ]:




