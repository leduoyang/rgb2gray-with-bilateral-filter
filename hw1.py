from scipy.spatial import distance
import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import exp , pow
import pdb


#function
###############################################################################
#Preprocessing
#conventional rgb2gray => get all weight combination => obtain the gray images according to the combinations
def readImg(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height , width , ch = img.shape
    plt.figure(0)
    plt.imshow(img)
    plt.show()    
    return img

def weightCombination_getNeightbors():
    list4w = []
    dic = {}
    n=0
    for i in range(11):
        remain = 10 - i 
        l,r= 0, 10
        while( l <= r):
            if (l+r) == remain :
                list4w.append([i,l,r])
                tem = str(i)+str(l)+str(r)
                dic[tem] = n
                n+=1
                if l != r :
                    list4w.append([i,r,l])
                    tem = str(i)+str(r)+str(l)
                    dic[tem] = n
                    n+=1
                l=l+1
                r=r-1
            elif (l+r) < remain :
                l=l+1
            elif (l+r) > remain :
                r=r-1
    list4w=np.array(list4w)
    list4w= list4w * 0.1
    D = distance.squareform(distance.pdist(list4w))
    dic = getNeighbors(D,list4w)
    return list4w,dic

def ObtainGrayImage(img,list4w):
    height , width , ch = img.shape
    L=len(list4w)
    gray_imgs = np.zeros((L,height,width), np.uint8)
    for i in range(L):
        w = list4w[i]
        tem = w[0]*img[:,:,0]+w[1]*img[:,:,1]+w[2]*img[:,:,2]
        gray_imgs[i,:,:] = tem    
    return gray_imgs

def conventional_rgb2gray(img):
    gray_img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    cv2.imwrite('2b_y.png', gray_img)
    plt.figure(1)
    plt.imshow(gray_img,cmp='gray')
    plt.show()
    
###############################################################################
#compute the ground truth for each sigma_s and sigma_r combination by bilateral filter => do the JBF with all the gray images as guidance images => compute the difference as cost between ground truth and JBF result 

def weighted_kernel(guide,sigma_s,sigma_r,channel): #kernel filtering
    if channel == 1:
         h ,w = guide.shape
    else:
        h ,w , ch= guide.shape
    kernel = np.ones((h,w))

    for i in range(h):
        for j in range(w):
            spatial_i , spatial_j = int(i - h // 2) , int(j - w // 2) #pixel considering - central pixel
            spatial_kernel = exp(-1 * ( pow(spatial_i,2) + pow(spatial_j,2) ) / ( 2 * pow(sigma_s,2) )) #hs
            
            if channel == 1: #single-channel
                range_kernel = exp(-1 * ( ( guide[i,j] - guide[h // 2,w // 2])**2 ) / ( 2 * sigma_r**2 )) #hr
            else: #color image
                range_kernel = exp(-1 * ( ( ( guide[i,j,0] - guide[h // 2,w // 2,0])**2 ) + ( ( guide[i,j,1] - guide[h // 2,w // 2,1])**2 ) + ( ( guide[i,j,2] - guide[h // 2,w // 2,2])**2 )  ) / ( 2 * sigma_r**2 )) #hr
            
            kernel[i,j] = spatial_kernel * range_kernel
            
    return kernel
 
def GroundTruth(src,sigma_s,sigma_r): #call bilateralFilter with all combination of sigma s & r
    height , width , ch = src.shape

    #BilateralFilter
    ground_truth = np.zeros((len(sigma_s)*len(sigma_r),height,width,ch), np.uint8)
    n=0
    for i in range(len(sigma_r)):
        sigmaColor = sigma_r[i] * 255
        for j in range(len(sigma_s)):
            sigmaSpace = sigma_s[j]
            
            d= 2*(3 * sigmaSpace)+1
            ground_truth[n,:,:,:] = bilateralFilter(src, d, sigmaColor, sigmaSpace) #cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)    
         
            n+=1    
    return ground_truth            

def bilateralFilter(src, d, sigmaColor, sigmaSpace):
    height , width , ch = src.shape
    G = np.zeros((height,width,ch), np.uint8)
    
    d= 2*(3 * sigmaSpace)+1 #window size
    border_size = d // 2
    f = cv2.copyMakeBorder(src, border_size, border_size, border_size, border_size,cv2.BORDER_REPLICATE) 
    
    for x in range(height):
        for y in range(width):
            box = f[x:x+d,y:y+d]
            guide = f[x:x+d,y:y+d]
            
            channel = 3 #color image
            kernel = weighted_kernel(guide,sigmaSpace,sigmaColor,channel)   
            
            G[x,y,0] = int((1/sum(sum(kernel)))*sum(sum(kernel * box[:,:,0])))
            G[x,y,1] = int((1/sum(sum(kernel)))*sum(sum(kernel * box[:,:,1])))
            G[x,y,2] = int((1/sum(sum(kernel)))*sum(sum(kernel * box[:,:,2])))       
    print("sigma_s : " + str(sigmaSpace) +",sigma_r : "+ str(sigmaColor))
    plt.figure(0)
    plt.imshow(G)
    plt.show()
    #pdb.set_trace()
    
    return G

def jointbilateralFilter(src,gray_imgs,sigma_s,sigma_r):
    height , width , ch = src.shape
    L = gray_imgs.shape
    L = L[0]
    result = np.zeros((len(sigma_s)*len(sigma_r),L,height,width,ch), np.uint8) # 9 combination from all the  spatial and range kernals, which each combination has 66 images
    n=0
    for i in range(len(sigma_r)):
        sigmaColor = sigma_r[i] * 255
        for j in range(len(sigma_s)):
            sigmaSpace = sigma_s[j]
    
            d= 2*(3 * sigmaSpace)+1
            border_size = d // 2
            f = cv2.copyMakeBorder(src, border_size, border_size, border_size, border_size,cv2.BORDER_REPLICATE)        
    
            for l in range(L):
                guide_img = gray_imgs[l,:,:]
                fp = cv2.copyMakeBorder(guide_img, border_size, border_size, border_size, border_size,cv2.BORDER_REPLICATE)       
    
    
                G = np.zeros((height,width,ch), np.uint8)
                
                
                for x in range(height):
                    for y in range(width):
                        box = f[x:x+d,y:y+d]
                        guide = fp[x:x+d,y:y+d]
                        
                        channel = 1 #single-channel
                        kernel = weighted_kernel(guide,sigmaSpace,sigmaColor,channel)
                        
                        G[x,y,0] = int((1/sum(sum(kernel)))*sum(sum(kernel * box[:,:,0])))
                        G[x,y,1] = int((1/sum(sum(kernel)))*sum(sum(kernel * box[:,:,1])))
                        G[x,y,2] = int((1/sum(sum(kernel)))*sum(sum(kernel * box[:,:,2])))
                result[n,l,:,:,:] = G
                print("sigma_s : " + str(sigmaSpace) +",sigma_r : "+ str(sigmaColor) +", no. gray image : " + str(l))
                plt.figure(0)
                plt.imshow(G)
                plt.show() 
            n+=1         
    return result

###############################################################################
    
#compute the cost for each weight combination => get the neighbors for each of them => identify who is the local minimum => voting , ranking
    
def computeCost(ground_truth,result): #between ground_truth and result after JBF
    Y = ground_truth.shape
    X = result.shape
    COST = np.zeros((Y[0],X[1]))
    
    for y in range(Y[0]):
        GT = ground_truth[y,:,:,:]
        for x in  range(X[1]):
            G = result[y,x,:,:,:]
            COST[y,x] = sum(sum(sum(abs(GT - G))))
    return COST

def getNeighbors(D,list4w): #for each weight combination
    dic = {}
    T = (3*(0.1**2)) ** 0.5
    for l in range(len(list4w)):
        tem = D[l,:]
        tem = tem <= T
        dic[l] = [i for i, x in enumerate(tem) if x]
    return dic

def isLocalMinimum(cost,l,neighbors): #identify whether the cost of weight combination now considering is local minimum
    neighbor_cost = [cost[x] for i,x in enumerate(neighbors)]
    tem = cost[l] <= neighbor_cost
    n = len([i for i, x in enumerate(tem) if x])
    l = len(neighbors)
    return n==l    

def voting(COST,dic):
    R,L = COST.shape
    vote=np.zeros((1,L))
    for r in range(R):
        for l in range(L):
            neighbors = dic[l]
            cost = COST[r,:]
            if isLocalMinimum(cost,l,neighbors):
                vote[0,l] +=1    
    return vote

def ranking_Output(vote,K):
    rank = (-vote).argsort() #[:n]
    for k in range(K):
        w = list4w[rank[0,k]]
        gray_img = gray_imgs[rank[0,k]]
        #print(w)
        #print("weight for rank "+str(k+1)+": "+w)
        cv2.imwrite('2b_y'+str(k+1)+'.png', gray_img)

        
###############################################################################

###############################################################################
if __name__ == "__main__":
    #pre-processing
    filename = './testdata/2b.png'
    img = readImg(filename)
    
    list4w,dic = weightCombination_getNeightbors()
    
    gray_imgs = ObtainGrayImage(img,list4w)
    
    #conventional rgb2gray    
    conventional_rgb2gray(img)
    
    #filter parameters
    src = img
    sigma_s = [1,2,3]
    sigma_r = [0.05,0.1,0.2]
    
    #processing
    print('start computing')
    
    ground_truth = GroundTruth(src,sigma_s,sigma_r) #BilateralFilter
    
    result = jointbilateralFilter(src,gray_imgs,sigma_s,sigma_r) #joint#BilateralFilter
    
    COST = computeCost(ground_truth,result) #compute cost
    
    vote = voting(COST,dic) #voting by local minimum identification
    
    K=3
    #ranking_Output(vote,k)
    rank = (-vote).argsort() #[:n]
    for k in range(K):
        w = list4w[rank[0,k]]
        gray_img = gray_imgs[rank[0,k]]
 
        print(w)
        #print("weight for rank "+str(k+1)+": "+w)
        cv2.imwrite('2b_y'+str(k+1)+'.png', gray_img)























































