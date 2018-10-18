#  Rgb2gary

implement rgb2gray selecting the best k gray images with bilateral , joint bilateral filter after local minimum selection and voting
(bilateral filter is treated as ground truth , compute the difference as cost with the joint bilateral filter result which take gray images with different weight combination as guidance )

## Bilateral filter 
Remove the noise while preserving the edge
(the kernels for filtering are about spatial and range information )

## Joint bilateral filter
take guidance image into account computing the weights for range kernel 

```
range kernel : if the pixel value in the kernel is close to the central one ,  the weight will be higher
spatial kernel : if the pixel index in the kernel is close to the central one ,  the weight will be higher
```

## Example

### original images
![alt text](https://github.com/leduoyang/rgb2gray-with-bilateral-filter/blob/master/img/2b.png)
![alt text](https://github.com/leduoyang/rgb2gray-with-bilateral-filter/blob/master/img/2c.png)


### conventional rgb2gray
![alt text](https://github.com/leduoyang/rgb2gray-with-bilateral-filter/blob/master/img/2b_y.png)
![alt text](https://github.com/leduoyang/rgb2gray-with-bilateral-filter/blob/master/img/2c_c.png)

### rgb2gray with BF,JBF
![alt text](https://github.com/leduoyang/rgb2gray-with-bilateral-filter/blob/master/img/2b_y1.png)
![alt text](https://github.com/leduoyang/rgb2gray-with-bilateral-filter/blob/master/img/2c_jbf.png)
