#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


# In[2]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os, shutil
import random
print("opencv version is",cv2.__version__)
print("numpy version is",np.__version__)

TP_50=0
TP_60=0
TP_70=0
TP_80=0
TP_90=0

FP_50=0
FP_60=0
FP_70=0
FP_80=0
FP_90=0

FN_50=0
FN_60=0
FN_70=0
FN_80=0
FN_90=0

bb_AP50 = []
bb_AP60 = []
bb_AP70 = []
bb_AP80 = []
bb_AP90 = []

con_AP50 = []

router = "%s/yolov4_MNIST_DVS512_10000.weights"%(os.getcwd())

#read .cfg .weights
net = cv2.dnn.readNetFromDarknet("/home/shon/code/MNIST_DVS_final/yolov4_MNIST_DVS512.cfg",router)
layer_names = net.getLayerNames()
#print(layer_names)
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
output_layers

#讀取分類檔、設定標記框顏色
classes = [line.strip() for line in open("/home/shon/code/MNIST_DVS_final/mnist.names")]
colors = [(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255)]

#先讀取要測試圖片
images_list = []
for img in os.listdir("/home/shon/code/MNIST_DVS_final/test_label"):
    if img.endswith(".jpg"):
        images_list.append(img.split(".")[0])

classes = [line.strip() for line in open("/home/shon/code/MNIST_DVS_final/mnist.names")]
colors = [(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(0,0,255)]
        
#開始偵測
for st in images_list:
    class_ids = []
    confidences = []
    boxes = []
    boxes2 = []
    img_st = "/home/shon/code/MNIST_DVS_final/test_label/%s.jpg"%(st)
    txt_st = "/home/shon/code/MNIST_DVS_final/test_label/%s.txt"%(st)
    print(img_st)
    print(txt_st)
    img = cv2.imread(img_st)
    img_og = cv2.imread(img_st)
    print("image size is",img.shape)
    
    #把寬、高、通道數拉出來
    height,width,channels = img.shape
    #影像送入模型做檢測
    blob = cv2.dnn.blobFromImage(img,1/255.0,(608,608),(0,0,0),True,crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    #偵測到幾個
    print("outs bbox is",len(outs),outs[0].shape)
    
    
    for out in outs:
        for detection in out:
            tx,ty,tw,th,confidence = detection[0:5]
            scores = detection[5:]
            class_id = np.argmax(scores)
            if confidence > 0.3:
                center_x = int(tx*width)
                center_y = int(ty*height)
                w = int(tw*width)
                h = int(th*height)
                #xmin ymin
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    print("box number",len(boxes))
   
    #非極大值抑制
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    f2 = open(txt_st,"r")
    #讀取txt放到boxes2
    for line in f2.readlines():
        cla,xt,yt,wt,ht = line.split(' ')
        boxes2.append([int(cla),float(xt),float(yt),float(wt),float(ht)])
    #shape[1]是高 shapep[0]是寬
    imgh=img.shape[0]
    imgw=img.shape[1]

    #YOLO轉成一般座標
    for j in range(len(boxes2)):
        cla,xt,yt,wt,ht = boxes2[j]
        wyo = wt*imgw
        hyo = ht*imgh
        xyo = ((xt*imgw)-(wyo/2))
        yyo = ((yt*imgh)-(hyo/2))
        boxes2[j]=[cla,int(xyo),int(yyo),int(wyo),int(hyo)]
    
    
    #計算所有IOU值
    final_iou = [0 for i in range(len(boxes2))]
    con_iou = [0 for i in range(len(boxes2))]
    for i in range(len(boxes2)):
        for j in range(len(boxes)):
            if j in indexes:
                cla,xt,yt,wt,ht = boxes2[i]
                x,y,w,h = boxes[j]
                rect1 = (xt,yt,xt+wt,yt+ht)
                rect2 = (x,y,x+w,y+h)
                iou = compute_iou(rect1, rect2)
                if iou > final_iou[i]:
                    final_iou[i]=iou
                    con_iou[i]=confidences[j]
        
    for i in range(len(boxes2)):
        print('iou : ',final_iou[i])
        print('con : ',con_iou[i])
        con_AP50.append(con_iou[i])
        
        if final_iou[i]>0.5:
            bb_AP50.append(1)
            TP_50+=1
        else:
            bb_AP50.append(0)
            if final_iou[i]>0:
              FP_50+=1
            else:
              FN_50+=1
            
        if final_iou[i]>0.6:
            bb_AP60.append(1)
            TP_60+=1
        else:
            bb_AP60.append(0)
            if final_iou[i]>0:
              FP_60+=1
            else:
              FN_60+=1
            
        if final_iou[i]>0.7:
            bb_AP70.append(1)
            TP_70+=1
        else:
            bb_AP70.append(0)
            if final_iou[i]>0:
              FP_70+=1
            else:
              FN_70+=1
            
        if final_iou[i]>0.8:
            bb_AP80.append(1)
            TP_80+=1
        else:
            bb_AP80.append(0)
            if final_iou[i]>0:
              FP_80+=1
            else:
              FN_80+=1
            
        if final_iou[i]>0.9:
            bb_AP90.append(1)
            TP_90+=1
        else:
            bb_AP90.append(0)
            if final_iou[i]>0:
              FP_90+=1
            else:
              FN_90+=1


    print("TP_50 :",TP_50)
    print("FP_50 :",FP_50)
    print("FN_50 :",FN_50)
    print("TP_60 :",TP_60)
    print("FP_60 :",FP_60)
    print("FN_60 :",FN_60)
    print("TP_70 :",TP_70)
    print("FP_70 :",FP_70)
    print("FN_70 :",FN_70)
    print("TP_80 :",TP_80)
    print("FP_80 :",FP_80)
    print("FN_80 :",FN_80)
    print("TP_90 :",TP_90)
    print("FP_90 :",FP_90)
    print("FN_90 :",FN_90)
    
    #存取檔案
    result_st = "%s/back10000/%s_og.jpg"%(os.getcwd(),st)
    result_st2 = "%s/back10000/%s_result.jpg"%(os.getcwd(),st)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            #(x,y)left top (x+w,y+h)right bottom
            cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
            #cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(img,label,(x,y-5),font,0.4,color,1,cv2.LINE_AA)
    for j in range(len(boxes2)):
        cla,xt,yt,wt,ht = boxes2[j]
        cv2.rectangle(img_og,(xt,yt),(xt+wt,yt+ht),color,1)
        
    cv2.imwrite(result_st, img_og)
    cv2.imwrite(result_st2, img)
    #存成tiff比較好，不容易模糊

router = "%s/back10000/result.txt"%(os.getcwd())
f = open(router,"w")
f.writelines("TP_50 :"+str(TP_50)+"\n")
f.writelines("FP_50 :"+str(FP_50)+"\n")
f.writelines("FN_50 :"+str(FN_50)+"\n")
f.writelines("TP_60 :"+str(TP_60)+"\n")
f.writelines("FP_60 :"+str(FP_60)+"\n")
f.writelines("FN_60 :"+str(FN_60)+"\n")
f.writelines("TP_70 :"+str(TP_70)+"\n")
f.writelines("FP_70 :"+str(FP_70)+"\n")
f.writelines("FN_70 :"+str(FN_70)+"\n")
f.writelines("TP_80 :"+str(TP_80)+"\n")
f.writelines("FP_80 :"+str(FP_80)+"\n")
f.writelines("FN_80 :"+str(FN_80)+"\n")
f.writelines("TP_90 :"+str(TP_90)+"\n")
f.writelines("FP_90 :"+str(FP_90)+"\n")
f.writelines("FN_90 :"+str(FN_90)+"\n")
     

precision = 0
precision = TP_50/(TP_50+FP_50)
print('precision_AP50 : ',precision)
f.writelines("precision_AP50 : "+str(precision)+"\n")
precision = TP_60/(TP_60+FP_60)
print('precision_AP60 : ',precision)
f.writelines("precision_AP60 : "+str(precision)+"\n")
precision = TP_70/(TP_70+FP_70)
print('precision_AP70 : ',precision)
f.writelines("precision_AP70 : "+str(precision)+"\n")
precision = TP_80/(TP_80+FP_80)
print('precision_AP80 : ',precision)
f.writelines("precision_AP80 : "+str(precision)+"\n")
precision = TP_90/(TP_90+FP_90)
print('precision_AP90 : ',precision)
f.writelines("precision_AP90 : "+str(precision)+"\n")



recall = 0
recall = TP_50/(TP_50+FN_50)
print('recall_AP50 : ',recall)
f.writelines("recall_AP50 : "+str(recall)+"\n")
recall = TP_60/(TP_60+FN_60)
print('recall_AP60 : ',recall)
f.writelines("recall_AP60 : "+str(recall)+"\n")
recall = TP_70/(TP_70+FN_70)
print('recall_AP70 : ',recall)
f.writelines("recall_AP70 : "+str(recall)+"\n")
recall = TP_80/(TP_80+FN_80)
print('recall_AP80 : ',recall)
f.writelines("recall_AP80 : "+str(recall)+"\n")
recall = TP_90/(TP_90+FN_90)
print('recall_AP90 : ',recall)
f.writelines("recall_AP90 : "+str(recall)+"\n")


f.close()
f2.close()


# In[ ]:





# In[2]:


import matplotlib
#matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

plt.figure(1,figsize=(6.4,4.8)) # 创建图表1
plt.title('Precision/Recall Curve')# give plot a title
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')
plt.axis([0,1,0,1])
 
#y_true和y_scores分别是gt label和predict score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(bb_AP50,con_AP50)
plt.figure(1)
plt.plot(recall,precision,'r',linewidth=4.0)
precision, recall, thresholds = precision_recall_curve(bb_AP60,con_AP50)
plt.plot(recall,precision,'b')
precision, recall, thresholds = precision_recall_curve(bb_AP70,con_AP50)
plt.plot(recall,precision,'g')
precision, recall, thresholds = precision_recall_curve(bb_AP80,con_AP50)
plt.plot(recall,precision,'y')
precision, recall, thresholds = precision_recall_curve(bb_AP90,con_AP50)
plt.plot(recall,precision,'c')
plt.legend(('AP50','AP60','AP70','AP80','AP90'),loc='lower left')

print(precision)
print(len(recall))
print(len(thresholds))

plt.savefig('p-rlast.tiff',dpi=200)
plt.show()


plt.figure(2,figsize=(6.4,4.8)) # 创建图表2
plt.title('Receiver Operating Characteristic Curve')# give plot a title
plt.xlabel('False Positive Rate')# make axis labels
plt.ylabel('True Positive Rate')
plt.axis([0,1,0,1])


plt.figure(2)
plt.plot([0,0,1],[0,1,1],'r',linewidth=4.0)
fpr, tpr, thresholds = roc_curve(bb_AP60,con_AP50)
plt.plot(fpr,tpr,'b')
print(thresholds)

fpr2, tpr2, thresholds = roc_curve(bb_AP70,con_AP50)
plt.plot(fpr2,tpr2,'g')
print(thresholds)

fpr3, tpr3, thresholds = roc_curve(bb_AP80,con_AP50)
plt.plot(fpr3,tpr3,'y')

fpr4, tpr4, thresholds = roc_curve(bb_AP90,con_AP50)
plt.plot(fpr4,tpr4,'c')
plt.plot([0, 1], [0, 1], 'k--') 

plt.legend(('perfect','AP60','AP70','AP80','AP90'),loc='lower right')

plt.savefig('ROC.tiff',dpi=200)
plt.show()


# In[3]:



print(len(tpr))


# In[4]:


plt.figure(3,figsize=(6.4,4.8)) # 创建图表2
plt.title('MNIST-DVS Detection Precision')# give plot a title
plt.xlabel('IOU Threshold')# make axis labels
plt.ylabel('Precision')
plt.axis([50,90,0,1])

plt.figure(3)
a,=plt.plot([50,60,70,80,90], [1,1,1,1,0.488],'y',linewidth=2.0)
b,=plt.plot([50,60,70,80,90], [1,1,0.907,0.574,0.093],'m',linewidth=2.0)
c,=plt.plot([50,60,70,80,90], [1,0.972,0.835,0.679,0.376],'c',linewidth=2.0)
d,=plt.plot([50,60,70,80,90], [1,0.941,0.765,0.588,0.147],'r',linewidth=2.0)
e,=plt.plot([50,60,70,80,90], [1,1,0.850,0.467,0.2],'g',linewidth=2.0)
f,=plt.plot([50,60,70,80,90], [1,0.852,0.759,0.556,0.130],'b',linewidth=2.0)
g,=plt.plot([50,60,70,80,90], [1,1,1,0.636,0.273],'m--',linewidth=2.0)
h,=plt.plot([50,60,70,80,90], [1,1,0.781,0.563,0.156],'c--',linewidth=2.0)
i,=plt.plot([50,60,70,80,90], [1,0.980,0.939,0.755,0.429],'b--',linewidth=2.0)
j,=plt.plot([50,60,70,80,90], [1,1,0.982,0.892,0.690],'g--',linewidth=2.0)


plt.legend([a,b,c,d,e,f,g,h,i,j],['zero','one','two','three','four','five','six','seven','eight','nine'],loc='lower left')

plt.savefig('precision.tiff',dpi=200)
plt.show()


# In[15]:


print(bb_AP50)

np.savez ('PR.npz', bb_AP50=bb_AP50,bb_AP60=bb_AP60,bb_AP70=bb_AP70,bb_AP80=bb_AP80,bb_AP90=bb_AP90, con_AP50=con_AP50)
npz = np.load('PR.npz')

te = []
te = npz['bb_AP50']
print(te)


# In[ ]:




