## 字节跳动面经题
- https://www.nowcoder.com/discuss/931198  
- # 如何用均匀分布构造正态分布
- https://ziyunge1999.github.io/blog/2020/09/06/constructNormalDistribution/
- 了解anchor-free?

1. Anchor base的优缺点
（1）使用anchor机制产生密集的anchor box，使得网络可直接在此基础上进行目标分类及边界框坐标回归；
（2）密集的anchor box可有效提高网络目标召回能力，对于小目标检测来说提升非常明显。
2. 缺点：
（1）anchor机制中，需要设定的超参：尺度(scale)和长宽比( aspect ratio)是比较难设计的。这需要较强的先验知识。
（2）冗余框非常之多：一张图像内的目标毕竟是有限的，基于每个anchor设定大量anchor box会产生大量的easy-sample，即完全不包含目标的背景框。这会造成正负样本严重不平衡问题，也是one-stage算法难以赶超two-stage算法的原因之一。
（3）网络实质上是看不见anchor box的，在anchor box的基础上进行边界回归更像是一种在范围比较小时候的强行记忆。
（4）基于anchor box进行目标类别分类时，IOU阈值超参设置也是一个问题，0.5？0.7？有同学可能也想到了CVPR2018的论文Cascade R-CNN，专门来讨论这个问题。感兴趣的同学可以移步：Naiyan Wang：CVPR18 Detection文章选介（上）
1. Anchor free 的缺点：
1）box的召回率不足。由于很多Anchor-free只在距离gt box中心较近的位置预测，导致回归box质量一般。比如YOLOV1。
2）对于重叠目标的检测效果一般。比如DenseBox系列。
2. 如何优化小目标检测：
2.1 为什么小目标的效果不好（难点在什么地方）：
- 小目标包含的特征少，分辨率低，背景信息较复杂、细节信息不明显以及定位精度要求较高。
- 卷积网络下采样率比较高
- 小目标映射到模型预测上的占比比较小
- 先验框的设置不合理
- 交并比的阈值不合理
2.2 如何提高小目标处理效果：
- 使用更好的骨干网络或者多尺度网络，提高特征提取的能力，或者引入注意力机制
- 采用更好的先验框生成算法
- 设置更低的交并比阈值，或者使用IOF替换IOU
- 优化损失函数采用GIOU，CIoU，DIOU，代替传统的IOU

2.3 NMS手写算法（与Soft-NMS）
def IoU(ground_truth, predictions):
    """
    ground_truth: [N,4]
    predictions: [M, 4]
    return: [N, M]
    """
    n, m = len(ground_truth), len(predictions)
    assert n > 0 and m > 0
    results = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            x1, y1, x2, y2 = ground_truth[i]
            x1_, y1_, x2_, y2_ = predictions[j]
            x1c, y1c, x2c, y2c = max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)
            if x2c < x1c or y2c < y1c: # 容易出现错误
                results[i][j] = 0.0
                continue
            inter, a1, a2 = (x2c - x1c) * (y2c - y1c), (x2 - x1) * (y2 - y1), (x2_ - x1_) * (y2_ - y1_)
            union = a1 + a2 - inter
            results[i][j] = inter / union
    return results


def NMS(boxes, scores, iou_thresh):
    """
    boxes: [N, 4]
    score: [N,]
    return: [M]
    """
    N = len(boxes)
    boxes = [[*boxes[i], scores[i]] for i in range(N)]
    boxes = sorted(boxes, key=lambda x: -x[-1])
    boxes_ = [b[:-1] for b in boxes]
    iou = IoU(boxes_, boxes_)
    remove_idx = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if remove_idx[i] == 0 and remove_idx[j] == 0 and iou[i][j] > iou_thresh:
                remove_idx[j] = 1
    boxes_ = [boxes_[i] for i in range(len(remove_idx)) if remove_idx[i] == 0]
    return boxes_
2.4 Soft-NMS：
[图片]
$$s_{i}=s_{i} e^{-\frac{\operatorname{iou}\left(\mathcal{M}, b_{i}\right)^{2}}{\sigma}}, \forall b_{i} \notin \mathcal{D}
$$
每次循环保留分数最大的box，然后降低和最大的box重叠度高的分数，最后得到一个重排之后的分数，保留前n个或者按照某个阈值保留大于阈值的所有box，删除小于其阈值的所有函数
3.1 YOLO的优缺点
优点
- 快速，pipline简单
- 背景误检率低
- 通用性强
但相比RCNN系列物体检测方法，YOLO具有以下缺点：
- 识别物体位置精准性差
- 召回率低
3.2 RetinaNet优缺点
优点：
- 通过修改loss函数实现处理了正负样本不均衡的问题
- 分析了One-Stage算法和Two Stage算法的差距，实现了一个精度可以媲美Two Stage算法的One Stage算法模型--RetinaNet
缺点：
- 速度太慢
3.3 Focal loss中的参数哪个关注困难样本，哪个解决长尾问题？
[图片]
$$\begin{aligned} \operatorname{CE}\left(p_{\mathrm{t}}\right) &=-\log \left(p_{\mathrm{t}}\right) \\
\operatorname{CE}\left(p_{\mathrm{t}}\right) &=-\alpha \log \left(p_{\mathrm{t}}\right) \\ \operatorname{FL}\left(p_{\mathrm{t}}\right) &=-\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right) \end{aligned}
$$
$$\alpha \in (0,1)
$$ 解决长尾问题（类别不均衡），$$\gamma$$解决困难样本。
当$$\alpha>0.5,
$$更加关注正样本，反之更加关注负样本
Focal Loss是解决前景-背景不平衡问题的一种常规解决方法。它侧重于硬前景样本的学习，并减少了简单背景样本的影响。
4. 旋转框检测的时候和水平框的区别
预测的内容不一样，水平框只需要价预测左上角和右下角的坐标，而旋转检测框需要预测多个点的坐标或者预测旋转角度等变量。
5. 介绍半监督方法
在【有标签数据+无标签数据】混合成的训练数据中使用的机器学习算法。一般假设，【无标签数据比有标签数据多】，甚至多得多。半监督学习的方法大都【建立在对数据的某种假设上】，只有满足这些假设，半监督算法才能有性能的保证，这也是限制了半监督学习应用的一大障碍。

1.【简单自训练（simple self-training）】：用有标签数据【训练一个分类器】，然后用这个分类器【对无标签数据进行分类】，这样就会产生伪标签（pseudo label）或软标签（soft label），挑选你认为分类正确的无标签样本（此处应该有一个【挑选准则】），把选出来的无标签样本用来训练分类器。
2.【协同训练（co-training）】：其实也是 self-training 的一种，但其思想是好的。假设每个数据可以从不同的角度（view）进行分类，不同角度可以训练出不同的分类器，然后【用这些从【不同角度】训练出来的分类器对无标签样本进行分类】，再选出认为可信的无标签样本加入训练集中。由于这些分类器从不同角度训练出来的，可以形成一种互补，而提高分类精度；就如同从不同角度可以更好地理解事物一样。
3.【半监督字典学习】：其实也是 self-training 的一种，【先是用有标签数据作为字典】，对无标签数据进行分类，挑选出你认为分类正确的无标签样本，【加入字典】中（此时的字典就变成了【半监督字典】了）

6. 常用的分类损失和常用的回归损失
损失函数的一般表示为 L(y,f(x))，用以衡量真实值 y和预测值 f(x)之间不一致的程度，一般越小越好。为了便于不同损失函数的比较，常将其表示为单变量的函数，在回归问题中这个变量为 [y-f(x)] ：残差表示，在分类问题中则为 yf(x) ： 趋势一致。
6.1 回归损失
平方损失 (squared loss) ：
$$(y-f(x))^2
$$
绝对值 (absolute loss) : 
$$|y-f(x)|
$$
Huber损失 (huber loss) : 
$$\left\{\begin{array}{cl}\frac{1}{2}[y-f(x)]^{2} & |y-f(x)| \leq \delta \\ \delta|y-f(x)|-\frac{1}{2} \delta^{2} & |y-f(x)|>\delta\end{array}\right.
$$
平方损失最常用，其缺点是对于异常点会施以较大的惩罚，因而不够robust。
绝对损失具有抵抗异常点干扰的特性，但是在y-f(x)处不连续可导，难以优化。
Huber损失是对二者的综合，当 |y-f(x)|小于一个事先指定的值 δ时，变为平方损失；大于δ时，则变成类似于绝对值损失，因此也是比较robust的损失函数

6.2 平方损失：
- 0-1损失 (zero-one loss):
0-1损失对每个错分类点都施以相同的惩罚，这样那些“错的离谱“(即 margin→∞) 的点并不会收到大的关注，这在直觉上不是很合适。另外0-1损失不连续、非凸，优化困难，因而常使用其他的代理损失函数进行优化。
- Logistic loss：
$$L(y,f(x))=log(1+e^{-yf(x)})
$$
- Hinge loss：
$$L(y,f(x))=max(0,1-yf(x))
$$
hinge loss为svm中使用的损失函数，hinge loss使得 yf(x)>1的样本损失皆为0，由此带来了稀疏解，使得svm仅通过少量的支持向量就能确定最终超平面。
- 指数损失(Exponential loss):
$$L(y,f(x))=e^{-yf(x)}
$$
exponential loss为AdaBoost中使用的损失函数，使用exponential loss能比较方便地利用加法模型推导出AdaBoost算法 。然而其和squared loss一样，对异常点敏感，不够robust。
- modified Huber loss:
$$
L(y, f(x))=\left\{\begin{array}{cl}
\max (0,1-y f(x))^{2} & \text { if } y f(x) \geq 1 \\
-4 y f(x) & \text { if } y f(x)<-1
\end{array}\right.

$$
modified huber loss结合了hinge loss和logistic loss的优点，既能在 yf(x)>1时产生稀疏解提高训练效率，又能进行概率估计。另外其对于  yf(x)<-1样本的惩罚以线性增加，这意味着受异常点的干扰较少，比较robust。
- 交叉熵损失：
$$L(y,f(x))=\frac{1}{N}\sum_iL_i=\frac{1}{N}\sum_i-[y_i*log(f(x_i))+(1-y_i)*log(1-f(x_i)]
$$
交叉熵损失函数求导：$$\frac{\partial L_{i}}{\partial w_{i}}=[\sigma(f(x_i))-y_i]*x_i
$$
我们重点关注$$[\sigma(f(x_i))-y_i]
$$，其的大小值反映了我们模型的错误程度，该值越大，说明模型效果越差，但是该值越大同时也会使得偏导值越大，从而模型学习速度更快。所以，使用逻辑函数得到概率，并结合交叉熵当损失函数时，在模型效果差的时候学习速度比较快，在模型效果好的时候学习速度变慢。

Deng [4]在2019年提出了ArcFace Loss，并在论文里说了Softmax Loss的两个缺点：1、随着分类数目的增大，分类层的线性变化矩阵参数也随着增大；2、对于封闭集分类问题，学习到的特征是可分离的，但对于开放集人脸识别问题，所学特征却没有足够的区分性。对于人脸识别问题，首先人脸数目(对应分类数目)是很多的，而且会不断有新的人脸进来，不是一个封闭集分类问题。
另外，sigmoid(softmax)+cross-entropy loss 擅长于学习类间的信息，因为它采用了类间竞争机制，它只关心对于正确标签预测概率的准确性，忽略了其他非正确标签的差异，导致学习到的特征比较散。基于这个问题的优化有很多，比如对softmax进行改进，如L-Softmax、SM-Softmax、AM-Softmax等。
7. anchor怎么设置，不同网络anchor设置的差别（SSD，faster-RCNN，yolo v3）
新手也能彻底搞懂的目标检测Anchor是什么?怎么科学设置?[附代码]_u010900574的博客-CSDN博客_神经网络anchor

8.  如果label中有错误的标签，但我们却不知道，怎样解决（当时答的是多个模型投票，没有得到回应，开放题）
标注数据存在错误怎么办?MIT& Google 提出用置信学习找出错误标注|附开源实现-极市开发者社区

9. 常见的将IOU有哪些？
（CIOU，DIOU，GIOU）

10. L1 loss, L2 loss, smooth L1 loss
- l1 loss:
$$L_1(y,f(x))=|y-f(x)|, L_1'=\pm f(x)
$$
- l2 loss:
$$L_2(y,f(x))=(y-f(x))^2,L_1'=2f(x)(f(x)-y)
$$
- Smooth l1:
$$\text { Smooth } \quad L_{1}=\begin{aligned}
&0.5 (x)^{2}, \quad|x|<1 \\
&|x|-0.5, \quad |x|>1
\end{aligned}
$$
$$\text { Smooth } \quad L_{1}=\begin{aligned}
&x, \quad|x|<1 \\
&-1, \quad x<-1 \\
&1, x>1
\end{aligned}
$$
根据fast rcnn的说法，"...... L1 loss that is less sensitive to outliers than the L2 loss used in R-CNN and SPPnet." 也就是smooth L1 loss让loss对于离群点更加鲁棒，即：相比于L2损失函数，其对离群点、异常值（outlier）不敏感，梯度变化相对更小，训练时不容易跑飞。
11. FPN和FSSD网络的结构：
[图片]
[图片]
12. 图像增强算法
- 直方图均衡化： 目的是为了增加对比度（早上7点左右拍摄的）但是会增加背景噪声。
- 双边滤波平滑直方图均衡化产生的噪声， 目的是保留边缘等高频信息（防鸟刺是针状物），平滑灰度相近的区域，综合考虑距离和颜色差异两种因素。
- 使用高提升滤波将使用拉普拉斯算子提取的细节与双边滤波平滑的图像融合。
- 最后将图像从hsv转到rbg图像。
13. 常用激活函数：
https://zhuanlan.zhihu.com/p/70810466
14. mixup：

15. 卷积输出宽度高度计算公式
https://zhuanlan.zhihu.com/p/29119239
$$H_{out} = (H_{in}-K_h+2*P_h)/S_h+1
$$
$$W_{out} = (W_{in}-K_w+2*P_w)/S_w+1
$$
Example: 
[图片]
From : Densely Connected Convolutional Networks
输入224*224， 第一层网络：K=7，S=2，输出高度和宽度为讲：$$(227-7+2*1)/2+1=112
$$
为什么选择7*7的卷积核，不选择3*3或者5*5的？
解答1：在这些尺寸中，如果尺寸太小，那么信息就丢失太严重，如果尺寸太大，信息的抽象层次不够高，计算量也更大，所以7*7的大小是一个最好的平衡。
16. attention的具体结构
